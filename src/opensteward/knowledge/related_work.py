"""Provider-independent related-work application orchestration."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge.hybrid_ranking import (
    KnowledgeHybridRankedMatch,
    KnowledgeHybridRankingMode,
    KnowledgeHybridRankingOptions,
    rank_knowledge_hybrid_corpus,
)
from opensteward.knowledge.lexical import (
    KnowledgeLexicalCorpus,
    KnowledgeLexicalQuery,
    build_knowledge_lexical_corpus,
)
from opensteward.knowledge.lexical_search import (
    MAX_KNOWLEDGE_LEXICAL_RESULTS,
    KnowledgeLexicalMatch,
    KnowledgeLexicalSearchOptions,
    search_knowledge_lexical_corpus,
)
from opensteward.knowledge.models import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)
from opensteward.knowledge.semantic_scoring import (
    KnowledgeSemanticScoringOptions,
    KnowledgeSemanticScoringResult,
    KnowledgeSemanticScoringStatus,
)

MAX_KNOWLEDGE_RELATED_WORK_RESULTS = 100

_RELATED_WORK_LEXICAL_FALLBACK_WARNING = (
    "Lexical candidate retrieval reached its safety limit; semantic and hybrid "
    "ranking were skipped to avoid incomplete score fusion."
)


class KnowledgeRelatedWorkMode(StrEnum):
    """Ranking paths available to the related-work application service."""

    LEXICAL_FALLBACK = "lexical_fallback"
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"


class KnowledgeRelatedWorkOptions(StrictKnowledgeModel):
    """Final-result and semantic-scoring bounds for related-work search."""

    max_results: int = Field(
        default=20,
        ge=1,
        le=MAX_KNOWLEDGE_RELATED_WORK_RESULTS,
    )
    semantic_scoring: KnowledgeSemanticScoringOptions = Field(
        default_factory=KnowledgeSemanticScoringOptions
    )


def _normalize_repository_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise ValueError("Repository paths must not be empty.")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError("Repository paths must be repository-relative.")

    parts = normalized.split("/")
    if any(not part for part in parts):
        raise ValueError("Repository paths must not contain empty segments.")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Repository paths must not contain '.' or '..' segments.")
    return normalized


def _validate_unique_strings(values: list[str], field_name: str) -> list[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    comparison_keys = [value.casefold() for value in values]
    if len(comparison_keys) != len(set(comparison_keys)):
        raise ValueError(f"{field_name} must be unique case-insensitively.")
    return values


class KnowledgeRelatedWorkItemSummary(StrictKnowledgeModel):
    """Concise historical item data returned with one related-work match."""

    reference: KnowledgeItemReference
    title: str = Field(min_length=1)
    summary: str | None = None
    url: str | None = None
    state: KnowledgeItemState
    decision_significance: DecisionSignificance
    updated_at: datetime
    labels: list[str]
    affected_paths: list[str]
    components: list[str]

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, updated_at: datetime) -> datetime:
        """Require an aware update time and normalize it to UTC."""

        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware.")
        return updated_at.astimezone(UTC)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        """Require unique, non-empty labels."""

        return _validate_unique_strings(labels, "Labels")

    @field_validator("affected_paths")
    @classmethod
    def normalize_affected_paths(cls, paths: list[str]) -> list[str]:
        """Normalize unique repository-relative paths."""

        normalized = [_normalize_repository_path(path) for path in paths]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Repository paths must be unique after normalization.")
        return normalized

    @field_validator("components")
    @classmethod
    def validate_components(cls, components: list[str]) -> list[str]:
        """Require unique, non-empty component names."""

        return _validate_unique_strings(components, "Components")

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the summarized source item key."""

        return self.reference.key


class KnowledgeRelatedWorkMatch(StrictKnowledgeModel):
    """One ranked related-work match with complete score provenance."""

    rank: int = Field(ge=1)
    mode: KnowledgeRelatedWorkMode
    item: KnowledgeRelatedWorkItemSummary
    lexical_match: KnowledgeLexicalMatch | None
    hybrid_match: KnowledgeHybridRankedMatch | None

    @model_validator(mode="after")
    def validate_match(self) -> Self:
        """Validate source identity and mode-specific ranking provenance."""

        if (
            self.lexical_match is not None
            and self.lexical_match.reference != self.item.reference
        ):
            raise ValueError("Lexical match must identify the summarized item.")
        if (
            self.hybrid_match is not None
            and self.hybrid_match.reference != self.item.reference
        ):
            raise ValueError("Hybrid match must identify the summarized item.")

        if self.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK:
            self._validate_lexical_fallback()
        elif self.mode == KnowledgeRelatedWorkMode.DETERMINISTIC:
            self._validate_deterministic()
        else:
            self._validate_hybrid()
        return self

    def _validate_lexical_fallback(self) -> None:
        if self.lexical_match is None or self.hybrid_match is not None:
            raise ValueError(
                "Lexical fallback requires a lexical match and no hybrid match."
            )

    def _validate_deterministic(self) -> None:
        if self.hybrid_match is None:
            raise ValueError("Deterministic related work requires a hybrid match.")
        if self.hybrid_match.mode != KnowledgeHybridRankingMode.DETERMINISTIC:
            raise ValueError(
                "Deterministic related work requires deterministic hybrid ranking."
            )
        if self.lexical_match != self.hybrid_match.lexical_match:
            raise ValueError(
                "Related-work lexical provenance must equal hybrid lexical provenance."
            )
        if self.hybrid_match.semantic_similarity is not None:
            raise ValueError(
                "Deterministic related work must not include semantic similarity."
            )

    def _validate_hybrid(self) -> None:
        if self.hybrid_match is None:
            raise ValueError("Hybrid related work requires a hybrid match.")
        if self.hybrid_match.mode != KnowledgeHybridRankingMode.HYBRID:
            raise ValueError("Hybrid related work requires hybrid ranking.")
        if self.lexical_match != self.hybrid_match.lexical_match:
            raise ValueError(
                "Related-work lexical provenance must equal hybrid lexical provenance."
            )
        if self.hybrid_match.semantic_similarity is None:
            raise ValueError("Hybrid related work requires semantic similarity.")

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the matched source item key."""

        return self.item.item_key

    @computed_field
    @property
    def score(self) -> int:
        """Return the authoritative public score for this ranking mode."""

        if self.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK:
            assert self.lexical_match is not None
            return self.lexical_match.score
        assert self.hybrid_match is not None
        return self.hybrid_match.score

    @computed_field
    @property
    def raw_score(self) -> int:
        """Return uncapped lexical points or fused weighted basis points."""

        if self.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK:
            assert self.lexical_match is not None
            return self.lexical_match.raw_score
        assert self.hybrid_match is not None
        return self.hybrid_match.total_weighted_basis_points

    @computed_field
    @property
    def lexical_evidence_count(self) -> int:
        """Return the number of retained lexical evidence records."""

        if self.lexical_match is None:
            return 0
        return len(self.lexical_match.evidence)

    @computed_field
    @property
    def contribution_count(self) -> int:
        """Return the number of fused ranking contributions."""

        if self.hybrid_match is None:
            return 0
        return len(self.hybrid_match.contributions)


def _hybrid_ranking_key(
    match: KnowledgeRelatedWorkMatch,
) -> tuple[int, int, int, str]:
    assert match.hybrid_match is not None
    semantic_score = (
        match.hybrid_match.semantic_similarity.score
        if match.hybrid_match.semantic_similarity is not None
        else 0
    )
    return (
        -match.hybrid_match.total_weighted_basis_points,
        -match.hybrid_match.core_lexical_raw_score,
        -semantic_score,
        match.item_key,
    )


class KnowledgeRelatedWorkResult(StrictKnowledgeModel):
    """Related-work matches with complete ranking and semantic provenance."""

    repository: KnowledgeRepositoryRef
    query: KnowledgeLexicalQuery
    options: KnowledgeRelatedWorkOptions
    as_of: datetime
    mode: KnowledgeRelatedWorkMode
    corpus_total_count: int = Field(ge=0)
    eligible_document_count: int = Field(ge=0)
    lexical_matched_document_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    semantic_status: KnowledgeSemanticScoringStatus | None = None
    semantic_provider: str | None = None
    semantic_model: str | None = None
    semantic_scored_document_count: int = Field(ge=0)
    matches: list[KnowledgeRelatedWorkMatch]
    warnings: list[str]

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, as_of: datetime) -> datetime:
        """Require an aware ranking time and normalize it to UTC."""

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        return as_of.astimezone(UTC)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        """Require non-empty unique warning text."""

        if any(not warning for warning in warnings):
            raise ValueError("Related-work warnings must be non-empty.")
        if len(warnings) != len(set(warnings)):
            raise ValueError("Related-work warnings must be unique.")
        return warnings

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate counts, ordering, identities, and mode-specific provenance."""

        if self.query.repository != self.repository:
            raise ValueError("Related-work query must belong to the result repository.")
        if self.corpus_total_count < self.eligible_document_count:
            raise ValueError("Corpus count must be at least the eligible count.")
        if self.eligible_document_count < self.lexical_matched_document_count:
            raise ValueError("Eligible count must be at least the lexical matched count.")
        if self.eligible_document_count < self.candidate_count:
            raise ValueError("Eligible count must be at least the candidate count.")
        if self.candidate_count < len(self.matches):
            raise ValueError("Candidate count must be at least the returned count.")
        if len(self.matches) > self.options.max_results:
            raise ValueError("Returned matches must not exceed options.max_results.")

        item_keys = [match.item_key for match in self.matches]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Related-work match keys must be unique.")
        if [match.rank for match in self.matches] != list(
            range(1, len(self.matches) + 1)
        ):
            raise ValueError("Related-work match ranks must be sequential from one.")
        if any(
            match.item.reference.repository != self.repository
            for match in self.matches
        ):
            raise ValueError("Every related-work match must belong to the repository.")
        if any(match.mode != self.mode for match in self.matches):
            raise ValueError("Every related-work match must use the result mode.")

        self._validate_ranking()
        if self.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK:
            self._validate_lexical_fallback()
        elif self.mode == KnowledgeRelatedWorkMode.DETERMINISTIC:
            self._validate_deterministic()
        else:
            self._validate_hybrid()
        return self

    def _validate_ranking(self) -> None:
        if self.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK:
            ranking = [
                (-match.raw_score, match.item_key)
                for match in self.matches
            ]
        else:
            ranking = [_hybrid_ranking_key(match) for match in self.matches]
        if ranking != sorted(ranking):
            raise ValueError("Related-work matches use an invalid ranking order.")

    def _validate_lexical_fallback(self) -> None:
        if (
            self.semantic_status is not None
            or self.semantic_provider is not None
            or self.semantic_model is not None
            or self.semantic_scored_document_count != 0
        ):
            raise ValueError(
                "Lexical fallback must not contain semantic provenance."
            )
        if self.warnings != [_RELATED_WORK_LEXICAL_FALLBACK_WARNING]:
            raise ValueError("Lexical fallback requires its exact coverage warning.")
        if self.candidate_count != self.lexical_matched_document_count:
            raise ValueError(
                "Lexical fallback candidate count must equal lexical matched count."
            )

    def _validate_deterministic(self) -> None:
        skipped_statuses = {
            KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
            KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
        }
        if self.semantic_status is not None and self.semantic_status not in skipped_statuses:
            raise ValueError(
                "Deterministic related work permits only skipped semantic statuses."
            )
        if (
            self.semantic_status
            == KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS
            and self.eligible_document_count != 0
        ):
            raise ValueError(
                "NO_ELIGIBLE_DOCUMENTS requires zero eligible related-work documents."
            )
        if (
            self.semantic_status == KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY
            and self.eligible_document_count == 0
        ):
            raise ValueError(
                "NO_SEMANTIC_QUERY requires eligible related-work documents."
            )
        if (
            self.semantic_provider is not None
            or self.semantic_model is not None
            or self.semantic_scored_document_count != 0
        ):
            raise ValueError(
                "Deterministic related work must not contain scored semantic provenance."
            )
        if self.warnings:
            raise ValueError("Deterministic related work must not contain warnings.")

    def _validate_hybrid(self) -> None:
        if self.semantic_status != KnowledgeSemanticScoringStatus.SCORED:
            raise ValueError("Hybrid related work requires SCORED semantic status.")
        if not self.semantic_provider or not self.semantic_model:
            raise ValueError("Hybrid related work requires provider and model provenance.")
        if self.semantic_scored_document_count != self.eligible_document_count:
            raise ValueError(
                "Hybrid semantic scoring must cover every eligible document."
            )
        if self.warnings:
            raise ValueError("Hybrid related work must not contain warnings.")
        for match in self.matches:
            assert match.hybrid_match is not None
            assert match.hybrid_match.semantic_similarity is not None
            if (
                match.hybrid_match.semantic_similarity.provider
                != self.semantic_provider
            ):
                raise ValueError(
                    "Hybrid match semantic provider must match result provenance."
                )
            if match.hybrid_match.semantic_similarity.model != self.semantic_model:
                raise ValueError(
                    "Hybrid match semantic model must match result provenance."
                )

    @computed_field
    @property
    def returned_count(self) -> int:
        """Return the number of related-work matches."""

        return len(self.matches)

    @computed_field
    @property
    def truncated(self) -> bool:
        """Return whether the final result bound omitted candidates."""

        return self.candidate_count > self.returned_count

    @computed_field
    @property
    def semantic_performed(self) -> bool:
        """Return whether complete semantic scoring was performed."""

        return self.semantic_status == KnowledgeSemanticScoringStatus.SCORED

    @computed_field
    @property
    def complete_ranking_coverage(self) -> bool:
        """Return whether candidate ranking had complete lexical coverage."""

        return self.mode != KnowledgeRelatedWorkMode.LEXICAL_FALLBACK


class KnowledgeSemanticScoringServiceProtocol(Protocol):
    """Injectable provider-neutral semantic-scoring service contract."""

    async def score(
        self,
        query: KnowledgeLexicalQuery,
        corpus: KnowledgeLexicalCorpus,
        *,
        options: KnowledgeSemanticScoringOptions | None = None,
    ) -> KnowledgeSemanticScoringResult:
        """Score every eligible corpus document or return a skipped result."""

        ...


def _validate_semantic_result(
    semantic_result: KnowledgeSemanticScoringResult,
    *,
    query: KnowledgeLexicalQuery,
    repository: KnowledgeRepositoryRef,
    eligible_document_count: int,
) -> None:
    if semantic_result.repository != repository:
        raise ValueError(
            "Semantic result repository must match the related-work repository."
        )
    if semantic_result.query != query:
        raise ValueError(
            "Semantic result query must match the related-work query."
        )
    if semantic_result.eligible_document_count != eligible_document_count:
        raise ValueError(
            "Semantic result eligible count must match related-work eligibility."
        )


def _build_item_summary(item: KnowledgeItem) -> KnowledgeRelatedWorkItemSummary:
    return KnowledgeRelatedWorkItemSummary(
        reference=item.to_reference(),
        title=item.title,
        summary=item.summary,
        url=item.url,
        state=item.state,
        decision_significance=item.decision_significance,
        updated_at=item.updated_at,
        labels=list(item.labels),
        affected_paths=list(item.affected_paths),
        components=list(item.components),
    )


def _build_lexical_matches(
    lexical_matches: list[KnowledgeLexicalMatch],
    items_by_key: dict[str, KnowledgeItem],
) -> list[KnowledgeRelatedWorkMatch]:
    return [
        KnowledgeRelatedWorkMatch(
            rank=rank,
            mode=KnowledgeRelatedWorkMode.LEXICAL_FALLBACK,
            item=_build_item_summary(items_by_key[lexical_match.item_key]),
            lexical_match=lexical_match,
            hybrid_match=None,
        )
        for rank, lexical_match in enumerate(lexical_matches, start=1)
    ]


def _build_hybrid_matches(
    hybrid_matches: list[KnowledgeHybridRankedMatch],
    items_by_key: dict[str, KnowledgeItem],
    mode: KnowledgeRelatedWorkMode,
) -> list[KnowledgeRelatedWorkMatch]:
    return [
        KnowledgeRelatedWorkMatch(
            rank=rank,
            mode=mode,
            item=_build_item_summary(items_by_key[hybrid_match.item_key]),
            lexical_match=hybrid_match.lexical_match,
            hybrid_match=hybrid_match,
        )
        for rank, hybrid_match in enumerate(hybrid_matches, start=1)
    ]


class KnowledgeRelatedWorkService:
    """Assemble bounded lexical, semantic, and hybrid related-work search."""

    def __init__(
        self,
        *,
        semantic_scoring_service: KnowledgeSemanticScoringServiceProtocol | None = None,
    ) -> None:
        self._semantic_scoring_service = semantic_scoring_service

    async def find(
        self,
        query: KnowledgeLexicalQuery,
        items: list[KnowledgeItem],
        *,
        as_of: datetime,
        options: KnowledgeRelatedWorkOptions | None = None,
    ) -> KnowledgeRelatedWorkResult:
        """Find related historical work with explicit ranking coverage."""

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        normalized_as_of = as_of.astimezone(UTC)
        effective_options = options or KnowledgeRelatedWorkOptions()

        if any(item.repository != query.repository for item in items):
            raise ValueError(
                "Every related-work item must belong to the query repository."
            )
        item_keys = [item.key for item in items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Related-work source item keys must be unique.")

        items_by_key = {item.key: item for item in items}
        corpus = build_knowledge_lexical_corpus(query.repository, items)
        lexical_result = search_knowledge_lexical_corpus(
            query,
            corpus,
            options=KnowledgeLexicalSearchOptions(
                max_results=MAX_KNOWLEDGE_LEXICAL_RESULTS,
                minimum_score=1,
            ),
        )
        if lexical_result.truncated:
            matches = _build_lexical_matches(
                lexical_result.matches[:effective_options.max_results],
                items_by_key,
            )
            return KnowledgeRelatedWorkResult(
                repository=query.repository,
                query=query,
                options=effective_options,
                as_of=normalized_as_of,
                mode=KnowledgeRelatedWorkMode.LEXICAL_FALLBACK,
                corpus_total_count=corpus.total_count,
                eligible_document_count=lexical_result.eligible_document_count,
                lexical_matched_document_count=(
                    lexical_result.matched_document_count
                ),
                candidate_count=lexical_result.matched_document_count,
                semantic_scored_document_count=0,
                matches=matches,
                warnings=[_RELATED_WORK_LEXICAL_FALLBACK_WARNING],
            )

        semantic_result: KnowledgeSemanticScoringResult | None = None
        semantic_scores = None
        if self._semantic_scoring_service is not None:
            semantic_result = await self._semantic_scoring_service.score(
                query,
                corpus,
                options=effective_options.semantic_scoring,
            )
            _validate_semantic_result(
                semantic_result,
                query=query,
                repository=query.repository,
                eligible_document_count=lexical_result.eligible_document_count,
            )
            if semantic_result.status == KnowledgeSemanticScoringStatus.SCORED:
                semantic_scores = semantic_result.similarities

        hybrid_result = rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            semantic_scores,
            as_of=normalized_as_of,
            options=KnowledgeHybridRankingOptions(
                max_results=effective_options.max_results,
                minimum_score=1,
            ),
        )
        mode = (
            KnowledgeRelatedWorkMode.HYBRID
            if hybrid_result.mode == KnowledgeHybridRankingMode.HYBRID
            else KnowledgeRelatedWorkMode.DETERMINISTIC
        )
        matches = _build_hybrid_matches(
            hybrid_result.matches,
            items_by_key,
            mode,
        )
        semantic_status = (
            semantic_result.status
            if semantic_result is not None
            else None
        )
        semantic_scored_document_count = (
            semantic_result.scored_document_count
            if semantic_result is not None
            and semantic_result.status == KnowledgeSemanticScoringStatus.SCORED
            else 0
        )
        return KnowledgeRelatedWorkResult(
            repository=query.repository,
            query=query,
            options=effective_options,
            as_of=normalized_as_of,
            mode=mode,
            corpus_total_count=corpus.total_count,
            eligible_document_count=lexical_result.eligible_document_count,
            lexical_matched_document_count=lexical_result.matched_document_count,
            candidate_count=hybrid_result.candidate_count,
            semantic_status=semantic_status,
            semantic_provider=(
                semantic_result.provider
                if semantic_result is not None
                else None
            ),
            semantic_model=(
                semantic_result.model
                if semantic_result is not None
                else None
            ),
            semantic_scored_document_count=semantic_scored_document_count,
            matches=matches,
            warnings=[],
        )
