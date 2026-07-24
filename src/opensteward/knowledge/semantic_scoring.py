"""Provider-neutral semantic-scoring orchestration for project knowledge."""

import unicodedata
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, computed_field, model_validator

from opensteward.knowledge.hybrid_ranking import KnowledgeSemanticSimilarity
from opensteward.knowledge.lexical import (
    KnowledgeLexicalCorpus,
    KnowledgeLexicalDocument,
    KnowledgeLexicalQuery,
)
from opensteward.knowledge.models import (
    KnowledgeItemReference,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS = 500
MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS = 20_000
MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS = 20_000
MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS = 1_000_000

_SEMANTIC_TRUNCATION_MARKER = "\n[truncated]"


class KnowledgeSemanticScoringError(ValueError):
    """Raised when local semantic-scoring orchestration cannot proceed safely."""


class KnowledgeSemanticScoringStatus(StrEnum):
    """Outcomes of provider-neutral semantic-scoring orchestration."""

    NO_ELIGIBLE_DOCUMENTS = "no_eligible_documents"
    NO_SEMANTIC_QUERY = "no_semantic_query"
    SCORED = "scored"


class KnowledgeSemanticScoringOptions(StrictKnowledgeModel):
    """Safety bounds for one semantic-scoring operation."""

    max_documents: int = Field(
        default=250,
        ge=1,
        le=MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS,
    )
    max_query_characters: int = Field(
        default=8_000,
        ge=100,
        le=MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS,
    )
    max_document_characters: int = Field(
        default=12_000,
        ge=100,
        le=MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS,
    )
    max_total_characters: int = Field(
        default=500_000,
        ge=100,
        le=MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS,
    )


class KnowledgeSemanticScoringDocument(StrictKnowledgeModel):
    """One bounded semantic document sent to an injected scorer."""

    reference: KnowledgeItemReference
    text: str = Field(
        min_length=1,
        max_length=MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS,
    )
    original_character_count: int = Field(
        ge=1,
        le=MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS,
    )
    emitted_character_count: int = Field(
        ge=1,
        le=MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS,
    )
    truncated: bool

    @model_validator(mode="after")
    def validate_character_metadata(self) -> Self:
        """Require exact character counts and truncation-marker provenance."""

        if self.emitted_character_count != len(self.text):
            raise ValueError("emitted_character_count must equal the document text length.")
        if self.original_character_count < self.emitted_character_count:
            raise ValueError(
                "original_character_count must not be less than emitted_character_count."
            )

        expected_truncated = (
            self.original_character_count > self.emitted_character_count
        )
        if self.truncated != expected_truncated:
            raise ValueError("Document truncation state must match its character counts.")
        if self.truncated and not self.text.endswith(_SEMANTIC_TRUNCATION_MARKER):
            raise ValueError("Truncated document text must end with the truncation marker.")
        if not self.truncated and self.text.endswith(_SEMANTIC_TRUNCATION_MARKER):
            raise ValueError(
                "Untruncated document text must not end with the truncation marker."
            )
        return self

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the source knowledge-item key."""

        return self.reference.key


class KnowledgeSemanticScoringRequest(StrictKnowledgeModel):
    """A complete bounded request for one semantic scorer invocation."""

    repository: KnowledgeRepositoryRef
    query_text: str = Field(
        min_length=1,
        max_length=MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS,
    )
    query_original_character_count: int = Field(
        ge=1,
        le=MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS,
    )
    query_emitted_character_count: int = Field(
        ge=1,
        le=MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS,
    )
    query_truncated: bool
    documents: list[KnowledgeSemanticScoringDocument] = Field(
        min_length=1,
        max_length=MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS,
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Validate query provenance, document identity, order, and total size."""

        if self.query_emitted_character_count != len(self.query_text):
            raise ValueError(
                "query_emitted_character_count must equal the query text length."
            )
        if self.query_original_character_count < self.query_emitted_character_count:
            raise ValueError(
                "query_original_character_count must not be less than "
                "query_emitted_character_count."
            )

        expected_truncated = (
            self.query_original_character_count
            > self.query_emitted_character_count
        )
        if self.query_truncated != expected_truncated:
            raise ValueError("Query truncation state must match its character counts.")
        if self.query_truncated and not self.query_text.endswith(
            _SEMANTIC_TRUNCATION_MARKER
        ):
            raise ValueError("Truncated query text must end with the truncation marker.")
        if not self.query_truncated and self.query_text.endswith(
            _SEMANTIC_TRUNCATION_MARKER
        ):
            raise ValueError(
                "Untruncated query text must not end with the truncation marker."
            )

        if any(
            document.reference.repository != self.repository
            for document in self.documents
        ):
            raise ValueError(
                "Every semantic document must belong to the request repository."
            )

        item_keys = self.document_item_keys
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Semantic request document keys must be unique.")
        if item_keys != sorted(item_keys):
            raise ValueError(
                "Semantic request documents must use item-key ascending order."
            )
        if self.total_request_characters > MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS:
            raise ValueError(
                "Semantic request exceeds the total-character safety limit."
            )
        return self

    @computed_field
    @property
    def document_count(self) -> int:
        """Return the number of documents in the request."""

        return len(self.documents)

    @computed_field
    @property
    def document_item_keys(self) -> list[str]:
        """Return request document keys in scorer input order."""

        return [document.item_key for document in self.documents]

    @computed_field
    @property
    def total_document_characters(self) -> int:
        """Return all emitted document characters."""

        return sum(document.emitted_character_count for document in self.documents)

    @computed_field
    @property
    def total_request_characters(self) -> int:
        """Return emitted query and document characters."""

        return len(self.query_text) + self.total_document_characters


class KnowledgeSemanticScorerScore(StrictKnowledgeModel):
    """One provider-normalized semantic score."""

    reference: KnowledgeItemReference
    score: int = Field(ge=0, le=100)

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the scored knowledge-item key."""

        return self.reference.key


class KnowledgeSemanticScorerResponse(StrictKnowledgeModel):
    """Provider provenance and normalized semantic scores."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    scores: list[KnowledgeSemanticScorerScore]

    @model_validator(mode="after")
    def validate_unique_scores(self) -> Self:
        """Reject duplicate score identities without imposing response order."""

        item_keys = [score.item_key for score in self.scores]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Semantic scorer response keys must be unique.")
        return self


class KnowledgeSemanticScorer(Protocol):
    """Adapter contract for an asynchronous semantic scorer."""

    async def score(
        self,
        request: KnowledgeSemanticScoringRequest,
    ) -> KnowledgeSemanticScorerResponse:
        """Return normalized semantic scores for every request document."""

        ...


class KnowledgeSemanticScoringResult(StrictKnowledgeModel):
    """Semantic-scoring outcome and complete provider provenance."""

    repository: KnowledgeRepositoryRef
    query: KnowledgeLexicalQuery
    status: KnowledgeSemanticScoringStatus
    eligible_document_count: int = Field(ge=0)
    scored_document_count: int = Field(ge=0)
    provider: str | None = None
    model: str | None = None
    query_truncated: bool
    truncated_document_count: int = Field(ge=0)
    emitted_character_count: int = Field(
        ge=0,
        le=MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS,
    )
    similarities: list[KnowledgeSemanticSimilarity]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Require status-specific counts, provenance, and complete identities."""

        if self.query.repository != self.repository:
            raise ValueError("Semantic result query must match the result repository.")
        if any(
            similarity.reference.repository != self.repository
            for similarity in self.similarities
        ):
            raise ValueError(
                "Every semantic similarity must belong to the result repository."
            )

        item_keys = [similarity.item_key for similarity in self.similarities]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Semantic result similarity keys must be unique.")
        if item_keys != sorted(item_keys):
            raise ValueError(
                "Semantic result similarities must use item-key ascending order."
            )

        if self.provider is not None and any(
            similarity.provider != self.provider
            for similarity in self.similarities
        ):
            raise ValueError("Every semantic similarity must match the result provider.")
        if self.model is not None and any(
            similarity.model != self.model
            for similarity in self.similarities
        ):
            raise ValueError("Every semantic similarity must match the result model.")
        if self.truncated_document_count > self.scored_document_count:
            raise ValueError(
                "truncated_document_count must not exceed scored_document_count."
            )

        if self.status == KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS:
            self._validate_no_eligible_documents()
        elif self.status == KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY:
            self._validate_no_semantic_query()
        else:
            self._validate_scored()
        return self

    def _validate_no_eligible_documents(self) -> None:
        if (
            self.eligible_document_count != 0
            or self.scored_document_count != 0
            or self.provider is not None
            or self.model is not None
            or self.similarities
            or self.query_truncated
            or self.truncated_document_count != 0
            or self.emitted_character_count != 0
        ):
            raise ValueError(
                "NO_ELIGIBLE_DOCUMENTS results must contain only zero-valued "
                "scoring metadata."
            )

    def _validate_no_semantic_query(self) -> None:
        if (
            self.eligible_document_count <= 0
            or self.scored_document_count != 0
            or self.provider is not None
            or self.model is not None
            or self.similarities
            or self.query_truncated
            or self.truncated_document_count != 0
            or self.emitted_character_count != 0
        ):
            raise ValueError(
                "NO_SEMANTIC_QUERY results require eligible documents and no "
                "scoring metadata."
            )

    def _validate_scored(self) -> None:
        if (
            self.eligible_document_count <= 0
            or self.scored_document_count != self.eligible_document_count
            or not self.provider
            or not self.model
            or len(self.similarities) != self.eligible_document_count
            or self.emitted_character_count <= 0
        ):
            raise ValueError(
                "SCORED results require complete coverage and provider provenance."
            )

    @computed_field
    @property
    def performed(self) -> bool:
        """Return whether the semantic scorer was invoked successfully."""

        return self.status == KnowledgeSemanticScoringStatus.SCORED

    @computed_field
    @property
    def complete_coverage(self) -> bool:
        """Return whether every eligible document has a semantic score."""

        return (
            self.status == KnowledgeSemanticScoringStatus.SCORED
            and self.scored_document_count == self.eligible_document_count
        )


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _identifier_comparison_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _stable_unique_identifiers(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _identifier_comparison_key(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _prepare_query_text(query: KnowledgeLexicalQuery) -> str:
    sections: list[str] = []
    if query.normalized_text is not None:
        sections.append(f"Text:\n{query.normalized_text}")
    if query.exact_phrases:
        phrases = "\n".join(_canonical_text(phrase) for phrase in query.exact_phrases)
        sections.append(f"Exact phrases:\n{phrases}")

    identifiers = _stable_unique_identifiers(
        [*query.identifiers, *query.text_identifiers]
    )
    if identifiers:
        sections.append(f"Identifiers:\n{'\n'.join(identifiers)}")
    return "\n\n".join(sections)


def _prepare_document_text(document: KnowledgeLexicalDocument) -> str:
    sections = [f"Title:\n{document.normalized_title}"]
    if document.normalized_summary is not None:
        sections.append(f"Summary:\n{document.normalized_summary}")
    if document.normalized_body is not None:
        sections.append(f"Body:\n{document.normalized_body}")
    return "\n\n".join(sections)


def _truncate_text(text: str, maximum_characters: int) -> tuple[str, bool]:
    if len(text) <= maximum_characters:
        return text, False

    prefix_length = maximum_characters - len(_SEMANTIC_TRUNCATION_MARKER)
    return (
        f"{text[:prefix_length]}{_SEMANTIC_TRUNCATION_MARKER}",
        True,
    )


def _eligible_documents(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
) -> list[KnowledgeLexicalDocument]:
    item_types = set(query.item_types)
    states = set(query.states)
    documents = [
        document
        for document in corpus.documents
        if document.reference.repository == query.repository
        and (
            not item_types
            or document.reference.item_type in item_types
        )
        and (not states or document.state in states)
    ]

    item_keys = [document.item_key for document in documents]
    if len(item_keys) != len(set(item_keys)):
        raise KnowledgeSemanticScoringError(
            "Eligible semantic document keys must be unique."
        )
    return sorted(documents, key=lambda document: document.item_key)


def _skipped_result(
    *,
    query: KnowledgeLexicalQuery,
    status: KnowledgeSemanticScoringStatus,
    eligible_document_count: int,
) -> KnowledgeSemanticScoringResult:
    return KnowledgeSemanticScoringResult(
        repository=query.repository,
        query=query,
        status=status,
        eligible_document_count=eligible_document_count,
        scored_document_count=0,
        query_truncated=False,
        truncated_document_count=0,
        emitted_character_count=0,
        similarities=[],
    )


class KnowledgeSemanticScoringService:
    """Prepare and validate one complete asynchronous semantic-scoring call."""

    def __init__(
        self,
        *,
        scorer: KnowledgeSemanticScorer,
    ) -> None:
        self._scorer = scorer

    async def score(
        self,
        query: KnowledgeLexicalQuery,
        corpus: KnowledgeLexicalCorpus,
        *,
        options: KnowledgeSemanticScoringOptions | None = None,
    ) -> KnowledgeSemanticScoringResult:
        """Score every eligible document or return an explicit skipped result."""

        if query.repository != corpus.repository:
            raise KnowledgeSemanticScoringError(
                "Semantic query and corpus repositories must match."
            )

        effective_options = options or KnowledgeSemanticScoringOptions()
        eligible_documents = _eligible_documents(query, corpus)
        if not eligible_documents:
            return _skipped_result(
                query=query,
                status=KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
                eligible_document_count=0,
            )
        if len(eligible_documents) > effective_options.max_documents:
            raise KnowledgeSemanticScoringError(
                "Eligible document count exceeds the semantic-scoring safety limit."
            )

        original_query_text = _prepare_query_text(query)
        if not original_query_text:
            return _skipped_result(
                query=query,
                status=KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
                eligible_document_count=len(eligible_documents),
            )
        if len(original_query_text) > MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS:
            raise KnowledgeSemanticScoringError(
                "Prepared semantic query exceeds the semantic-query safety limit."
            )

        query_text, query_truncated = _truncate_text(
            original_query_text,
            effective_options.max_query_characters,
        )
        scoring_documents = [
            self._prepare_scoring_document(document, effective_options)
            for document in eligible_documents
        ]
        total_characters = len(query_text) + sum(
            len(document.text) for document in scoring_documents
        )
        if total_characters > effective_options.max_total_characters:
            raise KnowledgeSemanticScoringError(
                "Prepared semantic request exceeds the total-character safety limit."
            )

        request = KnowledgeSemanticScoringRequest(
            repository=query.repository,
            query_text=query_text,
            query_original_character_count=len(original_query_text),
            query_emitted_character_count=len(query_text),
            query_truncated=query_truncated,
            documents=scoring_documents,
        )
        raw_response = await self._scorer.score(request)
        response = KnowledgeSemanticScorerResponse.model_validate(
            raw_response.model_dump(exclude_computed_fields=True)
            if isinstance(raw_response, KnowledgeSemanticScorerResponse)
            else raw_response
        )
        similarities = self._convert_response(request, response)
        return KnowledgeSemanticScoringResult(
            repository=query.repository,
            query=query,
            status=KnowledgeSemanticScoringStatus.SCORED,
            eligible_document_count=len(eligible_documents),
            scored_document_count=len(similarities),
            provider=response.provider,
            model=response.model,
            query_truncated=request.query_truncated,
            truncated_document_count=sum(
                document.truncated for document in request.documents
            ),
            emitted_character_count=request.total_request_characters,
            similarities=similarities,
        )

    @staticmethod
    def _prepare_scoring_document(
        document: KnowledgeLexicalDocument,
        options: KnowledgeSemanticScoringOptions,
    ) -> KnowledgeSemanticScoringDocument:
        original_text = _prepare_document_text(document)
        if len(original_text) > MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS:
            raise KnowledgeSemanticScoringError(
                "Prepared semantic document exceeds the semantic-document safety limit."
            )
        text, truncated = _truncate_text(
            original_text,
            options.max_document_characters,
        )
        return KnowledgeSemanticScoringDocument(
            reference=document.reference,
            text=text,
            original_character_count=len(original_text),
            emitted_character_count=len(text),
            truncated=truncated,
        )

    @staticmethod
    def _convert_response(
        request: KnowledgeSemanticScoringRequest,
        response: KnowledgeSemanticScorerResponse,
    ) -> list[KnowledgeSemanticSimilarity]:
        request_by_key = {
            document.item_key: document
            for document in request.documents
        }
        response_by_key = {
            score.item_key: score
            for score in response.scores
        }
        if response_by_key.keys() != request_by_key.keys():
            raise KnowledgeSemanticScoringError(
                "Semantic scorer response must exactly cover every request document."
            )

        for item_key, score in response_by_key.items():
            if score.reference != request_by_key[item_key].reference:
                raise KnowledgeSemanticScoringError(
                    "Semantic scorer response reference does not match the request "
                    "document."
                )

        return [
            KnowledgeSemanticSimilarity(
                reference=document.reference,
                score=response_by_key[document.item_key].score,
                provider=response.provider,
                model=response.model,
            )
            for document in request.documents
        ]
