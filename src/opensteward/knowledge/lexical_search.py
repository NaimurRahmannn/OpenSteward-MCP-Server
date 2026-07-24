"""Explainable deterministic lexical related-work search."""

import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge.lexical import (
    KnowledgeLexicalCorpus,
    KnowledgeLexicalDocument,
    KnowledgeLexicalField,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
)
from opensteward.knowledge.models import (
    DecisionSignificance,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MAX_KNOWLEDGE_LEXICAL_RESULTS = 100
MAX_KNOWLEDGE_LEXICAL_EVIDENCE_PER_MATCH = 500
MAX_KNOWLEDGE_LEXICAL_RAW_SCORE = 100_000
MAX_KNOWLEDGE_LEXICAL_SCORE = 100

KNOWLEDGE_TYPED_REFERENCE_POINTS = 60
KNOWLEDGE_UNTYPED_REFERENCE_POINTS = 50

KNOWLEDGE_TITLE_EXACT_PHRASE_POINTS = 30
KNOWLEDGE_SUMMARY_EXACT_PHRASE_POINTS = 20
KNOWLEDGE_BODY_EXACT_PHRASE_POINTS = 15

KNOWLEDGE_TITLE_IDENTIFIER_POINTS = 18
KNOWLEDGE_SUMMARY_IDENTIFIER_POINTS = 14
KNOWLEDGE_BODY_IDENTIFIER_POINTS = 10

KNOWLEDGE_EXACT_PATH_POINTS = 25
KNOWLEDGE_SHARED_DIRECTORY_POINTS = 8

KNOWLEDGE_COMPONENT_POINTS = 15
KNOWLEDGE_LABEL_POINTS = 10

KNOWLEDGE_TITLE_TERM_POINTS = 5
KNOWLEDGE_SUMMARY_TERM_POINTS = 3
KNOWLEDGE_BODY_TERM_POINTS = 1

class KnowledgeLexicalMatchKind(StrEnum):
    """Kinds of deterministic lexical evidence."""

    REFERENCE = "reference"
    EXACT_PHRASE = "exact_phrase"
    IDENTIFIER = "identifier"
    LABEL = "label"
    COMPONENT = "component"
    AFFECTED_PATH_EXACT = "affected_path_exact"
    AFFECTED_PATH_SHARED_DIRECTORY = "affected_path_shared_directory"
    TERM = "term"


class KnowledgeLexicalSearchOptions(StrictKnowledgeModel):
    """Bounds and threshold for one lexical corpus search."""

    max_results: int = Field(
        default=20,
        ge=1,
        le=MAX_KNOWLEDGE_LEXICAL_RESULTS,
    )
    minimum_score: int = Field(
        default=1,
        ge=1,
        le=MAX_KNOWLEDGE_LEXICAL_SCORE,
    )


class KnowledgeLexicalMatchEvidence(StrictKnowledgeModel):
    """One explainable contribution to a lexical match score."""

    kind: KnowledgeLexicalMatchKind
    field: KnowledgeLexicalField
    query_value: str = Field(min_length=1)
    document_value: str = Field(min_length=1)
    points: int = Field(ge=1, le=MAX_KNOWLEDGE_LEXICAL_SCORE)
    explanation: str = Field(min_length=1)


class KnowledgeLexicalMatch(StrictKnowledgeModel):
    """One historical item and its complete lexical match evidence."""

    reference: KnowledgeItemReference
    state: KnowledgeItemState
    decision_significance: DecisionSignificance
    updated_at: datetime
    evidence: list[KnowledgeLexicalMatchEvidence] = Field(
        min_length=1,
        max_length=MAX_KNOWLEDGE_LEXICAL_EVIDENCE_PER_MATCH,
    )

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, updated_at: datetime) -> datetime:
        """Require an aware update timestamp and normalize it to UTC."""

        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware.")
        return updated_at.astimezone(UTC)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Reject duplicate evidence identities and excessive raw scores."""

        identities = [
            (
                item.kind,
                item.field,
                _comparison_key(item.query_value),
                _comparison_key(item.document_value),
            )
            for item in self.evidence
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Lexical match evidence must not contain duplicates.")

        if sum(item.points for item in self.evidence) > MAX_KNOWLEDGE_LEXICAL_RAW_SCORE:
            raise ValueError("Lexical match raw score exceeds the safety limit.")
        return self

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the matched knowledge-item key."""

        return self.reference.key

    @computed_field
    @property
    def raw_score(self) -> int:
        """Return the uncapped sum of evidence points."""

        return sum(item.points for item in self.evidence)

    @computed_field
    @property
    def score(self) -> int:
        """Return the public lexical score capped at 100."""

        return min(MAX_KNOWLEDGE_LEXICAL_SCORE, self.raw_score)

    @computed_field
    @property
    def matched_fields(self) -> list[KnowledgeLexicalField]:
        """Return fields in their first evidence occurrence order."""

        return _stable_unique(
            (item.field for item in self.evidence),
            key=lambda field: field.value,
        )

    @computed_field
    @property
    def matched_kinds(self) -> list[KnowledgeLexicalMatchKind]:
        """Return match kinds in their first evidence occurrence order."""

        return _stable_unique(
            (item.kind for item in self.evidence),
            key=lambda kind: kind.value,
        )


class KnowledgeLexicalSearchResult(StrictKnowledgeModel):
    """Validated deterministic results for one lexical corpus search."""

    repository: KnowledgeRepositoryRef
    query: KnowledgeLexicalQuery
    corpus_total_count: int = Field(ge=0)
    eligible_document_count: int = Field(ge=0)
    matched_document_count: int = Field(ge=0)
    matches: list[KnowledgeLexicalMatch]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate repository identity, counts, uniqueness, and ranking."""

        if self.query.repository != self.repository:
            raise ValueError("Search query must belong to the result repository.")

        if any(
            match.reference.repository != self.repository
            for match in self.matches
        ):
            raise ValueError("Every lexical match must belong to the result repository.")

        item_keys = [match.item_key for match in self.matches]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Lexical search match keys must be unique.")

        if self.corpus_total_count < self.eligible_document_count:
            raise ValueError("Corpus count must be at least the eligible count.")
        if self.eligible_document_count < self.matched_document_count:
            raise ValueError("Eligible count must be at least the matched count.")
        if self.matched_document_count < len(self.matches):
            raise ValueError("Matched count must be at least the returned count.")
        if any(match.score < 1 for match in self.matches):
            raise ValueError("Returned lexical matches must have a positive score.")

        ranking = [
            (-match.raw_score, match.item_key)
            for match in self.matches
        ]
        if ranking != sorted(ranking):
            raise ValueError("Lexical matches must use deterministic ranking order.")
        return self

    @computed_field
    @property
    def returned_count(self) -> int:
        """Return the number of matches included in this result."""

        return len(self.matches)

    @computed_field
    @property
    def truncated(self) -> bool:
        """Return whether matching documents were omitted by the result bound."""

        return self.matched_document_count > self.returned_count


@dataclass(frozen=True)
class _EffectiveQuerySignals:
    references: list[KnowledgeLexicalReference]
    exact_phrases: list[tuple[str, str]]
    identifiers: list[tuple[str, str]]
    affected_paths: list[str]
    components: list[tuple[str, str]]
    labels: list[tuple[str, str]]
    terms: list[str]


def _normalize_nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _comparison_key(value: str) -> str:
    return _normalize_nfkc(value).casefold()


def _canonical_text(value: str) -> str:
    return " ".join(_comparison_key(value).split())


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


def _prepare_preserved_strings(
    values: Iterable[str],
    *,
    canonicalizer: Callable[[str], str],
) -> list[tuple[str, str]]:
    prepared = [
        (value, canonicalizer(value))
        for value in values
    ]
    return _stable_unique(prepared, key=lambda item: item[1])


def _prepare_query_signals(
    query: KnowledgeLexicalQuery,
) -> _EffectiveQuerySignals:
    references = _stable_unique(
        [*query.references, *query.text_references],
        key=lambda reference: reference.key,
    )
    exact_phrases = _prepare_preserved_strings(
        query.exact_phrases,
        canonicalizer=_canonical_text,
    )
    identifiers = _prepare_preserved_strings(
        [*query.identifiers, *query.text_identifiers],
        canonicalizer=_comparison_key,
    )
    components = _prepare_preserved_strings(
        query.components,
        canonicalizer=_comparison_key,
    )
    labels = _prepare_preserved_strings(
        query.labels,
        canonicalizer=_comparison_key,
    )
    affected_paths = _stable_unique(
        query.affected_paths,
        key=lambda path: path,
    )
    return _EffectiveQuerySignals(
        references=references,
        exact_phrases=exact_phrases,
        identifiers=identifiers,
        affected_paths=affected_paths,
        components=components,
        labels=labels,
        terms=list(query.text_terms),
    )


def _append_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    item: KnowledgeLexicalMatchEvidence,
) -> None:
    if len(evidence) >= MAX_KNOWLEDGE_LEXICAL_EVIDENCE_PER_MATCH:
        raise ValueError(
            "Lexical match evidence exceeds the configured safety limit."
        )
    evidence.append(item)


def _add_reference_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> None:
    for reference in signals.references:
        if reference.external_id != document.reference.external_id:
            continue
        if (
            reference.item_type is not None
            and reference.item_type != document.reference.item_type
        ):
            continue

        typed = reference.item_type is not None
        _append_evidence(
            evidence,
            KnowledgeLexicalMatchEvidence(
                kind=KnowledgeLexicalMatchKind.REFERENCE,
                field=KnowledgeLexicalField.EXTERNAL_ID,
                query_value=reference.key,
                document_value=(
                    f"{document.reference.item_type.value}:"
                    f"{document.reference.external_id}"
                ),
                points=(
                    KNOWLEDGE_TYPED_REFERENCE_POINTS
                    if typed
                    else KNOWLEDGE_UNTYPED_REFERENCE_POINTS
                ),
                explanation=(
                    "Explicit typed reference matched this historical item."
                    if typed
                    else "Explicit item reference matched this historical item."
                ),
            ),
        )


def _add_exact_phrase_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> None:
    fields = (
        (
            KnowledgeLexicalField.TITLE,
            document.normalized_title,
            KNOWLEDGE_TITLE_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item title.",
        ),
        (
            KnowledgeLexicalField.SUMMARY,
            document.normalized_summary,
            KNOWLEDGE_SUMMARY_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item summary.",
        ),
        (
            KnowledgeLexicalField.BODY,
            document.normalized_body,
            KNOWLEDGE_BODY_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item body.",
        ),
    )
    for query_value, canonical_phrase in signals.exact_phrases:
        for field, document_text, points, explanation in fields:
            if document_text is not None and canonical_phrase in document_text:
                _append_evidence(
                    evidence,
                    KnowledgeLexicalMatchEvidence(
                        kind=KnowledgeLexicalMatchKind.EXACT_PHRASE,
                        field=field,
                        query_value=query_value,
                        document_value=canonical_phrase,
                        points=points,
                        explanation=explanation,
                    ),
                )
                break


def _first_identifier_match(
    query_identifier: str,
    document_identifiers: list[str],
) -> str | None:
    return next(
        (
            identifier
            for identifier in document_identifiers
            if _comparison_key(identifier) == query_identifier
        ),
        None,
    )


def _add_identifier_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> None:
    fields = (
        (
            KnowledgeLexicalField.TITLE,
            document.title_identifiers,
            KNOWLEDGE_TITLE_IDENTIFIER_POINTS,
            "Identifier matched the historical item title.",
        ),
        (
            KnowledgeLexicalField.SUMMARY,
            document.summary_identifiers,
            KNOWLEDGE_SUMMARY_IDENTIFIER_POINTS,
            "Identifier matched the historical item summary.",
        ),
        (
            KnowledgeLexicalField.BODY,
            document.body_identifiers,
            KNOWLEDGE_BODY_IDENTIFIER_POINTS,
            "Identifier matched the historical item body.",
        ),
    )
    for query_value, canonical_identifier in signals.identifiers:
        for field, document_identifiers, points, explanation in fields:
            document_value = _first_identifier_match(
                canonical_identifier,
                document_identifiers,
            )
            if document_value is not None:
                _append_evidence(
                    evidence,
                    KnowledgeLexicalMatchEvidence(
                        kind=KnowledgeLexicalMatchKind.IDENTIFIER,
                        field=field,
                        query_value=query_value,
                        document_value=document_value,
                        points=points,
                        explanation=explanation,
                    ),
                )
                break


def _shared_directory_depth(first_path: str, second_path: str) -> int:
    first_directories = first_path.split("/")[:-1]
    second_directories = second_path.split("/")[:-1]
    depth = 0
    for first, second in zip(
        first_directories,
        second_directories,
        strict=False,
    ):
        if first != second:
            break
        depth += 1
    return depth


def _best_shared_directory_path(
    query_path: str,
    document_paths: list[str],
) -> str | None:
    candidates = [
        (-_shared_directory_depth(query_path, document_path), document_path)
        for document_path in document_paths
        if _shared_directory_depth(query_path, document_path) >= 2
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def _add_path_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> None:
    document_paths = document.affected_paths
    for query_path in signals.affected_paths:
        if query_path in document_paths:
            _append_evidence(
                evidence,
                KnowledgeLexicalMatchEvidence(
                    kind=KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
                    field=KnowledgeLexicalField.AFFECTED_PATH,
                    query_value=query_path,
                    document_value=query_path,
                    points=KNOWLEDGE_EXACT_PATH_POINTS,
                    explanation=(
                        "Repository path exactly matched historical changed-path evidence."
                    ),
                ),
            )
            continue

        document_path = _best_shared_directory_path(query_path, document_paths)
        if document_path is not None:
            _append_evidence(
                evidence,
                KnowledgeLexicalMatchEvidence(
                    kind=(
                        KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY
                    ),
                    field=KnowledgeLexicalField.AFFECTED_PATH,
                    query_value=query_path,
                    document_value=document_path,
                    points=KNOWLEDGE_SHARED_DIRECTORY_POINTS,
                    explanation=(
                        "Repository paths share a specific directory hierarchy."
                    ),
                ),
            )


def _add_structured_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    *,
    signals: list[tuple[str, str]],
    document_values: list[str],
    kind: KnowledgeLexicalMatchKind,
    field: KnowledgeLexicalField,
    points: int,
    explanation: str,
) -> None:
    document_value_set = set(document_values)
    for query_value, canonical_value in signals:
        if canonical_value not in document_value_set:
            continue
        _append_evidence(
            evidence,
            KnowledgeLexicalMatchEvidence(
                kind=kind,
                field=field,
                query_value=query_value,
                document_value=canonical_value,
                points=points,
                explanation=explanation,
            ),
        )


def _add_term_evidence(
    evidence: list[KnowledgeLexicalMatchEvidence],
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> None:
    fields = (
        (
            KnowledgeLexicalField.TITLE,
            set(document.title_terms),
            KNOWLEDGE_TITLE_TERM_POINTS,
            "Term matched the historical item title.",
        ),
        (
            KnowledgeLexicalField.SUMMARY,
            set(document.summary_terms),
            KNOWLEDGE_SUMMARY_TERM_POINTS,
            "Term matched the historical item summary.",
        ),
        (
            KnowledgeLexicalField.BODY,
            set(document.body_terms),
            KNOWLEDGE_BODY_TERM_POINTS,
            "Term matched the historical item body.",
        ),
    )
    for term in signals.terms:
        for field, document_terms, points, explanation in fields:
            if term in document_terms:
                _append_evidence(
                    evidence,
                    KnowledgeLexicalMatchEvidence(
                        kind=KnowledgeLexicalMatchKind.TERM,
                        field=field,
                        query_value=term,
                        document_value=term,
                        points=points,
                        explanation=explanation,
                    ),
                )
                break


def _score_document(
    signals: _EffectiveQuerySignals,
    document: KnowledgeLexicalDocument,
) -> KnowledgeLexicalMatch | None:
    evidence: list[KnowledgeLexicalMatchEvidence] = []
    _add_reference_evidence(evidence, signals, document)
    _add_exact_phrase_evidence(evidence, signals, document)
    _add_identifier_evidence(evidence, signals, document)
    _add_path_evidence(evidence, signals, document)
    _add_structured_evidence(
        evidence,
        signals=signals.components,
        document_values=document.components,
        kind=KnowledgeLexicalMatchKind.COMPONENT,
        field=KnowledgeLexicalField.COMPONENT,
        points=KNOWLEDGE_COMPONENT_POINTS,
        explanation="Component matched historical component evidence.",
    )
    _add_structured_evidence(
        evidence,
        signals=signals.labels,
        document_values=document.labels,
        kind=KnowledgeLexicalMatchKind.LABEL,
        field=KnowledgeLexicalField.LABEL,
        points=KNOWLEDGE_LABEL_POINTS,
        explanation="Label matched historical label evidence.",
    )
    _add_term_evidence(evidence, signals, document)

    if not evidence:
        return None
    return KnowledgeLexicalMatch(
        reference=document.reference,
        state=document.state,
        decision_significance=document.decision_significance,
        updated_at=document.updated_at,
        evidence=evidence,
    )


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


def search_knowledge_lexical_corpus(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
    *,
    options: KnowledgeLexicalSearchOptions | None = None,
) -> KnowledgeLexicalSearchResult:
    """Search a prepared corpus using explainable deterministic lexical evidence."""

    if query.repository != corpus.repository:
        raise ValueError("Lexical query and corpus repositories must match.")

    effective_options = options or KnowledgeLexicalSearchOptions()
    signals = _prepare_query_signals(query)
    eligible_documents = [
        document
        for document in corpus.documents
        if _is_eligible(query, document)
    ]
    matches: list[KnowledgeLexicalMatch] = []
    for document in eligible_documents:
        match = _score_document(signals, document)
        if (
            match is not None
            and match.score >= effective_options.minimum_score
        ):
            matches.append(match)

    matches.sort(key=lambda match: (-match.raw_score, match.item_key))
    matched_document_count = len(matches)
    return KnowledgeLexicalSearchResult(
        repository=corpus.repository,
        query=query,
        corpus_total_count=corpus.total_count,
        eligible_document_count=len(eligible_documents),
        matched_document_count=matched_document_count,
        matches=matches[:effective_options.max_results],
    )
