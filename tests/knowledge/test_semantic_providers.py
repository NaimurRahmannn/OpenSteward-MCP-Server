"""Tests for local embedding scoring and optional Groq reranking."""

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from opensteward.knowledge import (
    KnowledgeItemReference,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSemanticScoringDocument,
    KnowledgeSemanticScoringRequest,
    KnowledgeSourceKind,
)
from opensteward.knowledge.semantic_providers import (
    GroqSemanticReranker,
    LocalEmbeddingSemanticScorer,
)
from opensteward.knowledge.semantic_settings import (
    SemanticSettings,
)

REPOSITORY = KnowledgeRepositoryRef(
    provider="github",
    namespace="acme",
    name="framework",
)


def settings(
    **updates: Any,
) -> SemanticSettings:
    """Build semantic settings without reading the environment."""

    values: dict[str, Any] = {
        "_env_file": None,
        "semantic_enabled": True,
        "embedding_model": "test-embedding",
        "embedding_max_tokens": 512,
        "embedding_batch_size": 4,
        "semantic_max_documents": 100,
        "groq_enabled": False,
    }
    values.update(updates)
    return SemanticSettings(**values)


def document(
    external_id: str,
    text: str,
) -> KnowledgeSemanticScoringDocument:
    """Build one semantic document."""

    reference = KnowledgeItemReference(
        repository=REPOSITORY,
        item_type=KnowledgeItemType.ISSUE,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.GITHUB,
        title=f"Issue {external_id}",
    )

    return KnowledgeSemanticScoringDocument(
        reference=reference,
        text=text,
        original_character_count=len(text),
        emitted_character_count=len(text),
        truncated=False,
    )


def request() -> KnowledgeSemanticScoringRequest:
    """Build a stable two-document semantic request."""

    query_text = "parser registry architecture"

    return KnowledgeSemanticScoringRequest(
        repository=REPOSITORY,
        query_text=query_text,
        query_original_character_count=len(query_text),
        query_emitted_character_count=len(query_text),
        query_truncated=False,
        documents=[
            document("1", "parser registry design"),
            document("2", "unrelated documentation"),
        ],
    )


class FakeEmbeddingBackend:
    """Return deterministic vectors and record bounded batching."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.document_calls: list[
            tuple[list[str], int]
        ] = []

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        self.document_calls.append(
            (texts, batch_size)
        )

        return [
            (
                [1.0, 0.0]
                if "parser" in text
                else [0.0, 1.0]
            )
            for text in texts
        ]


@pytest.mark.asyncio
async def test_local_embedding_scorer_returns_complete_normalized_scores() -> None:
    backend = FakeEmbeddingBackend()
    scorer = LocalEmbeddingSemanticScorer(
        settings=settings(),
        backend=backend,
    )

    response = await scorer.score(request())

    assert response.provider == "fastembed"
    assert response.model == "test-embedding"
    assert [
        score.score
        for score in response.scores
    ] == [100, 0]
    assert backend.queries == [
        "parser registry architecture"
    ]
    assert backend.document_calls == [
        (
            [
                "parser registry design",
                "unrelated documentation",
            ],
            4,
        )
    ]


@pytest.mark.asyncio
async def test_groq_reranks_only_top_candidates_with_strict_schema() -> None:
    captured: list[httpx.Request] = []
    semantic_request = request()
    selected_key = (
        semantic_request.documents[0].item_key
    )

    def handler(
        http_request: httpx.Request,
    ) -> httpx.Response:
        captured.append(http_request)

        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "scores": [
                                        {
                                            "item_key": (
                                                selected_key
                                            ),
                                            "score": 50,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    configured = settings(
        groq_enabled=True,
        groq_api_key=SecretStr("test-secret"),
        groq_model="test-groq",
        groq_max_candidates=1,
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        reranker = GroqSemanticReranker(
            settings=configured,
            client=client,
        )
        scorer = LocalEmbeddingSemanticScorer(
            settings=configured,
            backend=FakeEmbeddingBackend(),
            groq_reranker=reranker,
        )

        response = await scorer.score(
            semantic_request
        )

    assert response.provider == "fastembed+groq"
    assert response.model == (
        "test-embedding+test-groq"
    )
    assert [
        score.score
        for score in response.scores
    ] == [85, 0]
    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assert payload["response_format"][
        "json_schema"
    ]["strict"] is True
    user_payload = json.loads(
        payload["messages"][1]["content"]
    )
    assert len(user_payload["items"]) == 1
    assert (
        user_payload["items"][0]["item_key"]
        == selected_key
    )
    assert (
        captured[0].headers["authorization"]
        == "Bearer test-secret"
    )


@pytest.mark.asyncio
async def test_groq_failure_falls_back_to_local_scores() -> None:
    transport = httpx.MockTransport(
        lambda http_request: httpx.Response(
            status_code=429,
            headers={"Retry-After": "1"},
            json={"error": "rate limited"},
        )
    )
    configured = settings(
        groq_enabled=True,
        groq_api_key=SecretStr("test-secret"),
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        response = await LocalEmbeddingSemanticScorer(
            settings=configured,
            backend=FakeEmbeddingBackend(),
            groq_reranker=GroqSemanticReranker(
                settings=configured,
                client=client,
            ),
        ).score(request())

    assert response.provider == "fastembed"
    assert [
        score.score
        for score in response.scores
    ] == [100, 0]
