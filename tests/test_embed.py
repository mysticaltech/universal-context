"""Tests for embedding module and semantic/hybrid search."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_artifact,
    semantic_search,
    store_embedding,
)
from universal_context.db.schema import apply_schema
from universal_context.embed import (
    EmbedProvider,
    LocalEmbedProvider,
    OpenAIEmbedProvider,
    create_embed_provider,
)

# --- Fixtures ---


@pytest.fixture
async def db(tmp_path):
    d = UCDatabase(f"mem://test_embed_{id(tmp_path)}")
    await d.connect()
    await apply_schema(d)
    yield d
    await d.close()


def _fake_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic fake embedding based on text hash for testing."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    values = []
    for i in range(dim):
        byte_idx = i % len(h)
        values.append((h[byte_idx] + i) % 256 / 255.0)
    return values


def _mock_embed_provider(dim: int = 768) -> MagicMock:
    """Create a mock EmbedProvider with embed_document and embed_query."""
    provider = MagicMock(spec=EmbedProvider)
    provider.dim = dim
    provider.embed_document = AsyncMock(return_value=[0.1] * dim)
    provider.embed_query = AsyncMock(return_value=[0.1] * dim)
    return provider


# --- create_embed_provider tests ---


class TestCreateEmbedProvider:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=False)
        result = await create_embed_provider(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_local_provider_default(self):
        """Default provider is local (EmbeddingGemma)."""
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="local")
        result = await create_embed_provider(config)
        assert result is not None
        assert isinstance(result, LocalEmbedProvider)
        assert result.dim == 768

    @pytest.mark.asyncio
    async def test_auto_returns_local(self):
        """Auto provider returns local (always available, no API keys)."""
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="auto")
        result = await create_embed_provider(config)
        assert result is not None
        assert isinstance(result, LocalEmbedProvider)

    @pytest.mark.asyncio
    async def test_openai_provider(self):
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = await create_embed_provider(config)
        assert result is not None
        assert isinstance(result, OpenAIEmbedProvider)
        assert result.dim == 1536

    @pytest.mark.asyncio
    async def test_openrouter_provider(self):
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="openrouter")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            result = await create_embed_provider(config)
        assert result is not None
        assert isinstance(result, OpenAIEmbedProvider)

    @pytest.mark.asyncio
    async def test_openai_no_key_returns_none(self):
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="openai")
        with patch.dict(os.environ, {}, clear=True):
            result = await create_embed_provider(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_openrouter_no_key_returns_none(self):
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="openrouter")
        with patch.dict(os.environ, {}, clear=True):
            result = await create_embed_provider(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_has_embed_methods(self):
        """All providers expose embed_query and embed_document."""
        from universal_context.config import UCConfig

        config = UCConfig(use_llm=True, embed_provider="local")
        provider = await create_embed_provider(config)
        assert hasattr(provider, "embed_query")
        assert hasattr(provider, "embed_document")
        assert hasattr(provider, "dim")


# --- store_embedding + semantic_search tests ---


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_embedding(self, db):
        """Store an embedding on an artifact, verify it's stored."""
        aid = await create_artifact(db, kind="summary", content="JWT auth middleware")

        emb = _fake_embedding("JWT auth middleware")
        await store_embedding(db, aid, emb)

        result = await db.query(f"SELECT embedding FROM {aid}")
        assert result
        assert result[0]["embedding"] is not None
        assert len(result[0]["embedding"]) == 768

    @pytest.mark.asyncio
    async def test_knn_search_embedded_fallback(self, db):
        """Vector search on embedded uses brute-force cosine fallback."""
        contents = [
            "Fixed JWT token expiration in auth middleware",
            "Added pagination to the users API endpoint",
            "Refactored database migration system",
        ]
        for c in contents:
            aid = await create_artifact(db, kind="summary", content=c)
            await store_embedding(db, aid, _fake_embedding(c))

        query_emb = _fake_embedding("JWT authentication token bug")
        results = await semantic_search(db, query_emb, kind="summary", limit=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_knn_search_empty_db(self, db):
        """KNN search on empty DB returns empty list."""
        query_emb = _fake_embedding("anything")
        results = await semantic_search(db, query_emb, limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_knn_search_filters_by_kind(self, db):
        """Semantic search with kind filter on embedded returns only matching kind."""
        aid1 = await create_artifact(db, kind="summary", content="auth fix")
        await store_embedding(db, aid1, _fake_embedding("auth fix"))

        aid2 = await create_artifact(db, kind="transcript", content="raw transcript")
        await store_embedding(db, aid2, _fake_embedding("raw transcript"))

        results = await semantic_search(
            db, _fake_embedding("auth"), kind="summary", limit=5,
        )
        assert isinstance(results, list)
        # All results should be of kind "summary"
        for r in results:
            assert r["kind"] == "summary"


# --- Summarizer with embed provider tests ---


class TestSummarizerEmbedding:
    @pytest.mark.asyncio
    async def test_summarizer_embeds_after_summary(self, db):
        """When embed provider is given, summarizer stores embedding on summary."""
        from universal_context.daemon.processors.summarizer import TurnSummarizer
        from universal_context.db.queries import (
            claim_next_job,
            create_run,
            create_scope,
            create_turn_with_artifact,
        )

        scope = await create_scope(db, "test")
        run = await create_run(db, str(scope["id"]), "claude")
        await create_turn_with_artifact(
            db, str(run["id"]), 1, "test message", "User: test\nAssistant: done"
        )

        job = await claim_next_job(db)

        provider = _mock_embed_provider()
        summarizer = TurnSummarizer(embed_fn=provider)
        result = await summarizer.process(db, job)

        assert result["embedded"] is True
        provider.embed_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summarizer_handles_embed_failure(self, db):
        """When embed provider raises, summarizer still succeeds."""
        from universal_context.daemon.processors.summarizer import TurnSummarizer
        from universal_context.db.queries import (
            claim_next_job,
            create_run,
            create_scope,
            create_turn_with_artifact,
        )

        scope = await create_scope(db, "test")
        run = await create_run(db, str(scope["id"]), "claude")
        await create_turn_with_artifact(
            db, str(run["id"]), 1, "test", "User: test\nAssistant: ok"
        )

        job = await claim_next_job(db)

        provider = _mock_embed_provider()
        provider.embed_document = AsyncMock(side_effect=RuntimeError("API down"))
        summarizer = TurnSummarizer(embed_fn=provider)
        result = await summarizer.process(db, job)

        assert result["method"] == "extractive"
        assert result["embedded"] is False

    @pytest.mark.asyncio
    async def test_summarizer_skips_embed_when_none(self, db):
        """When no embed provider, summarizer skips embedding."""
        from universal_context.daemon.processors.summarizer import TurnSummarizer
        from universal_context.db.queries import (
            claim_next_job,
            create_run,
            create_scope,
            create_turn_with_artifact,
        )

        scope = await create_scope(db, "test")
        run = await create_run(db, str(scope["id"]), "claude")
        await create_turn_with_artifact(
            db, str(run["id"]), 1, "test", "User: test\nAssistant: ok"
        )

        job = await claim_next_job(db)
        summarizer = TurnSummarizer()  # no embed provider
        result = await summarizer.process(db, job)

        assert result["embedded"] is False
