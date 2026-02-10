"""Integration tests using real LLM and embedding providers.

These tests make actual API calls and load real models.
They verify the full system works end-to-end, not just mocked interfaces.

Run:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "embed"     # just embedding tests
    pytest tests/test_integration.py -v -k "llm"        # just LLM tests
    pytest tests/test_integration.py -v -k "pipeline"   # full pipeline

Skip in CI:
    pytest -m "not integration"

Requires:
    - OpenRouter API key in ~/.uc/config.yaml or OPENROUTER_API_KEY env var
    - ~600MB disk for EmbeddingGemma model (auto-downloaded on first run)
"""

from __future__ import annotations

import math

import pytest

from universal_context.config import UCConfig
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_artifact,
    create_run,
    create_scope,
    create_turn_with_artifact,
    store_embedding,
)
from universal_context.db.schema import apply_schema
from universal_context.embed import LocalEmbedProvider, create_embed_provider
from universal_context.llm import create_llm_fn

# ---------------------------------------------------------------------------
# Markers & skip conditions
# ---------------------------------------------------------------------------

_config = UCConfig.load()
HAS_LLM_KEY = _config.get_api_key("openrouter") is not None

pytestmark = pytest.mark.integration

skip_no_key = pytest.mark.skipif(
    not HAS_LLM_KEY, reason="No OpenRouter API key configured",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embed_provider():
    """Module-scoped local embed provider (model loads once for all tests)."""
    return LocalEmbedProvider()


@pytest.fixture
async def db():
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Local Embedding Tests
# ---------------------------------------------------------------------------


class TestLocalEmbedding:
    """Tests with real EmbeddingGemma-300M model."""

    async def test_embed_query_dimensions(self, embed_provider):
        """embed_query returns a vector with the correct dimension."""
        vec = await embed_provider.embed_query("authentication bug fix")
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    async def test_embed_document_dimensions(self, embed_provider):
        """embed_document returns a vector with the correct dimension."""
        vec = await embed_provider.embed_document("Fixed JWT token expiration in auth middleware")
        assert len(vec) == 768

    async def test_embedding_similarity_related_texts(self, embed_provider):
        """Related texts should have higher cosine similarity than unrelated."""
        auth_query = await embed_provider.embed_query("authentication login security")
        auth_doc = await embed_provider.embed_document(
            "Fixed JWT token validation in the login endpoint"
        )
        unrelated_doc = await embed_provider.embed_document(
            "Refactored CSS grid layout for the dashboard sidebar"
        )

        sim_related = _cosine_sim(auth_query, auth_doc)
        sim_unrelated = _cosine_sim(auth_query, unrelated_doc)

        assert sim_related > sim_unrelated, (
            f"Related similarity ({sim_related:.4f}) should be > "
            f"unrelated similarity ({sim_unrelated:.4f})"
        )

    async def test_asymmetric_prefixes_differ(self, embed_provider):
        """Query and document embeddings for the same text should differ (asymmetric model)."""
        text = "database migration system"
        query_vec = await embed_provider.embed_query(text)
        doc_vec = await embed_provider.embed_document(text)

        # They should be similar but NOT identical (different prefixes)
        sim = _cosine_sim(query_vec, doc_vec)
        assert 0.5 < sim < 1.0, f"Expected high but non-identical similarity, got {sim:.4f}"

    async def test_deterministic_output(self, embed_provider):
        """Same input produces same output (no randomness)."""
        text = "reproducibility test"
        vec1 = await embed_provider.embed_document(text)
        vec2 = await embed_provider.embed_document(text)
        assert vec1 == vec2

    async def test_different_inputs_differ(self, embed_provider):
        """Different inputs produce different embeddings."""
        vec_a = await embed_provider.embed_document("python async database")
        vec_b = await embed_provider.embed_document("french pastry baking")
        sim = _cosine_sim(vec_a, vec_b)
        assert sim < 0.95, f"Unrelated texts too similar: {sim:.4f}"

    async def test_batch_consistency(self, embed_provider):
        """Multiple sequential calls produce valid, consistent results."""
        texts = [
            "Set up the authentication module with JWT tokens",
            "Add rate limiting to the login endpoint",
            "Write unit tests for the auth middleware",
            "Deploy the service to production",
            "Refactor database connection pooling",
        ]
        vecs = [await embed_provider.embed_document(t) for t in texts]
        assert all(len(v) == 768 for v in vecs)

        # Auth-related texts (0, 1, 2) should cluster vs unrelated (3, 4)
        auth_sims = [_cosine_sim(vecs[0], vecs[i]) for i in (1, 2)]
        other_sims = [_cosine_sim(vecs[0], vecs[i]) for i in (3, 4)]
        avg_auth = sum(auth_sims) / len(auth_sims)
        avg_other = sum(other_sims) / len(other_sims)
        assert avg_auth > avg_other, (
            f"Auth cluster avg ({avg_auth:.4f}) should be > other avg ({avg_other:.4f})"
        )


# ---------------------------------------------------------------------------
# Embedding + DB Storage Tests
# ---------------------------------------------------------------------------


class TestEmbeddingStorage:
    """Test storing real embeddings in SurrealDB and retrieving them."""

    async def test_store_real_embedding(self, db, embed_provider):
        """Store a real embedding vector on an artifact."""
        content = "Fixed JWT token expiration in auth middleware"
        vec = await embed_provider.embed_document(content)

        aid = await create_artifact(db, kind="summary", content=content)
        await store_embedding(db, aid, vec)

        result = await db.query(f"SELECT embedding FROM {aid}")
        assert result
        stored_vec = result[0]["embedding"]
        assert len(stored_vec) == 768
        # Verify roundtrip fidelity (first few values should match)
        for i in range(10):
            assert abs(stored_vec[i] - vec[i]) < 1e-6

    async def test_store_multiple_embeddings(self, db, embed_provider):
        """Store multiple real embeddings and verify they're distinct."""
        contents = [
            "JWT authentication token validation",
            "CSS grid layout refactoring",
            "Database migration error handling",
        ]

        aids = []
        vecs = []
        for content in contents:
            vec = await embed_provider.embed_document(content)
            aid = await create_artifact(db, kind="summary", content=content)
            await store_embedding(db, aid, vec)
            aids.append(aid)
            vecs.append(vec)

        # Retrieve and verify each one
        for aid, original_vec in zip(aids, vecs):
            result = await db.query(f"SELECT embedding FROM {aid}")
            stored_vec = result[0]["embedding"]
            sim = _cosine_sim(stored_vec, original_vec)
            assert sim > 0.9999, f"Roundtrip fidelity lost: {sim:.6f}"

    async def test_brute_force_cosine_search(self, db, embed_provider):
        """Manual cosine similarity search on embedded DB (no HNSW needed)."""
        contents = [
            "Fixed JWT token expiration in auth middleware",
            "Added pagination to the users API endpoint",
            "Refactored database migration system",
            "Updated CSS styles for the login form",
            "Implemented rate limiting for API routes",
        ]

        aids = []
        for content in contents:
            vec = await embed_provider.embed_document(content)
            aid = await create_artifact(db, kind="summary", content=content)
            await store_embedding(db, aid, vec)
            aids.append(aid)

        # Search for auth-related content
        query_vec = await embed_provider.embed_query("authentication security login")

        # Brute-force: fetch all embeddings and rank by cosine
        results = await db.query("SELECT id, content, embedding FROM artifact")
        ranked = []
        for r in results:
            if r.get("embedding"):
                sim = _cosine_sim(query_vec, r["embedding"])
                ranked.append((sim, str(r["id"]), r["content"]))
        ranked.sort(reverse=True)

        # Top result should be auth-related
        assert "auth" in ranked[0][2].lower() or "login" in ranked[0][2].lower(), (
            f"Expected auth-related top result, got: {ranked[0][2]}"
        )


# ---------------------------------------------------------------------------
# LLM Summarization Tests
# ---------------------------------------------------------------------------


@skip_no_key
class TestLLMSummarization:
    """Tests with real OpenRouter API calls."""

    async def test_create_llm_fn_returns_callable(self):
        """create_llm_fn returns a working callable with real config."""
        config = UCConfig.load()
        llm_fn = await create_llm_fn(config)
        assert llm_fn is not None

    async def test_summarize_short_transcript(self):
        """LLM produces a summary from a short transcript."""
        config = UCConfig.load()
        llm_fn = await create_llm_fn(config)

        transcript = (
            "User: Set up JWT authentication for the API\n"
            "Assistant: I'll create an auth module with JWT tokens. "
            "I've set up login and verify endpoints in auth.py using "
            "python-jose for token generation with HS256 signing.\n"
        )

        summary = await llm_fn(transcript)
        assert isinstance(summary, str)
        assert len(summary) > 10, f"Summary too short: {summary!r}"
        assert len(summary) < len(transcript) * 3, "Summary unreasonably long"

    async def test_summarize_multi_turn_transcript(self):
        """LLM summarizes a multi-turn conversation transcript."""
        config = UCConfig.load()
        llm_fn = await create_llm_fn(config)

        transcript = (
            "User: We need to add rate limiting to our API\n"
            "Assistant: I'll implement rate limiting using a sliding window "
            "algorithm. I've added a RateLimiter class in middleware.py that "
            "tracks requests per IP using an in-memory counter with TTL.\n\n"
            "User: Can you make it configurable per endpoint?\n"
            "Assistant: Updated the rate limiter to accept per-route configuration. "
            "Now you can set different limits via decorators: @rate_limit(10, '1m') "
            "for 10 requests per minute. Default is 60/min for all routes.\n\n"
            "User: Add Redis support for distributed deployments\n"
            "Assistant: Replaced the in-memory store with a Redis backend. "
            "Uses MULTI/EXEC for atomic increment+expire. Falls back to "
            "in-memory if Redis is unavailable. Added RATE_LIMIT_REDIS_URL "
            "env var for configuration.\n"
        )

        summary = await llm_fn(transcript)
        assert isinstance(summary, str)
        assert len(summary) > 20

    async def test_llm_model_is_grok(self):
        """Verify config loads the correct model."""
        config = UCConfig.load()
        assert config.llm_model == "x-ai/grok-4.1-fast"
        assert config.llm_provider == "openrouter"


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


@skip_no_key
class TestFullPipeline:
    """End-to-end tests with real LLM + real embeddings."""

    async def test_summarizer_with_real_providers(self, db, embed_provider):
        """TurnSummarizer uses real LLM for summary + real embeddings for storage."""
        from universal_context.daemon.processors.summarizer import TurnSummarizer
        from universal_context.db.queries import claim_next_job

        config = UCConfig.load()
        llm_fn = await create_llm_fn(config)

        # Set up a turn with a transcript
        scope = await create_scope(db, "integration-test")
        run = await create_run(db, str(scope["id"]), "claude")
        await create_turn_with_artifact(
            db,
            str(run["id"]),
            sequence=1,
            user_message="Set up JWT auth",
            raw_content=(
                "User: Set up JWT authentication for the API\n"
                "Assistant: I've created auth.py with JWT token generation "
                "using HS256 signing. The login endpoint validates credentials "
                "and returns a signed token. The verify middleware checks the "
                "token signature and expiration on protected routes."
            ),
        )

        job = await claim_next_job(db)
        assert job is not None

        # Run with real LLM + real embeddings
        summarizer = TurnSummarizer(llm_fn=llm_fn, embed_fn=embed_provider)
        result = await summarizer.process(db, job)

        assert result["method"] == "llm", f"Expected LLM summary, got {result['method']}"
        assert result["embedded"] is True
        assert result["length"] > 10

        # Verify the summary artifact has a real embedding
        summary_id = result["summary_id"]
        stored = await db.query(f"SELECT content, embedding FROM {summary_id}")
        assert stored
        assert len(stored[0]["content"]) > 10
        assert len(stored[0]["embedding"]) == 768

    async def test_multi_turn_pipeline(self, db, embed_provider):
        """Full pipeline: multiple turns → LLM summaries → embeddings → cosine search."""
        from universal_context.daemon.processors.summarizer import TurnSummarizer
        from universal_context.db.queries import claim_next_job

        config = UCConfig.load()
        llm_fn = await create_llm_fn(config)

        scope = await create_scope(db, "multi-turn-test", "/tmp/test-project")
        run = await create_run(db, str(scope["id"]), "claude")

        transcripts = [
            (
                "Fix the auth bug",
                "User: The JWT tokens are expiring too early\n"
                "Assistant: Found the issue — the expiration was set to 60 seconds "
                "instead of 60 minutes. Fixed in auth.py line 42. Also added a "
                "config option for TOKEN_EXPIRY_MINUTES.",
            ),
            (
                "Add user roles",
                "User: Add role-based access control\n"
                "Assistant: Implemented RBAC with three roles: admin, editor, viewer. "
                "Added a roles table, updated the JWT payload to include role claims, "
                "and created a @require_role decorator for endpoint protection.",
            ),
            (
                "Optimize database queries",
                "User: The dashboard is slow, profile the DB queries\n"
                "Assistant: Found N+1 query in the user list endpoint. Replaced with "
                "a single JOIN query. Added an index on users.created_at. Response time "
                "dropped from 2.3s to 45ms.",
            ),
        ]

        for seq, (msg, transcript) in enumerate(transcripts, 1):
            await create_turn_with_artifact(
                db, str(run["id"]), sequence=seq,
                user_message=msg, raw_content=transcript,
            )

        # Process all 3 jobs
        summarizer = TurnSummarizer(llm_fn=llm_fn, embed_fn=embed_provider)
        results = []
        for _ in range(3):
            job = await claim_next_job(db)
            assert job is not None
            result = await summarizer.process(db, job)
            results.append(result)

        # All should be LLM-summarized and embedded
        assert all(r["method"] == "llm" for r in results)
        assert all(r["embedded"] is True for r in results)

        # Brute-force cosine search: "authentication token bug" should match turn 1
        query_vec = await embed_provider.embed_query("authentication token bug fix")
        summaries = await db.query(
            "SELECT id, content, embedding FROM artifact WHERE kind = 'summary'"
        )

        ranked = []
        for s in summaries:
            if s.get("embedding"):
                sim = _cosine_sim(query_vec, s["embedding"])
                ranked.append((sim, s["content"]))
        ranked.sort(reverse=True)

        assert len(ranked) == 3
        # Top result should mention auth/JWT/token
        top_content = ranked[0][1].lower()
        assert any(term in top_content for term in ("auth", "jwt", "token", "expir")), (
            f"Expected auth-related top result, got: {ranked[0][1][:100]}"
        )

    async def test_create_embed_provider_from_config(self):
        """create_embed_provider returns a real local provider from config."""
        config = UCConfig.load()
        provider = await create_embed_provider(config)
        assert provider is not None
        assert isinstance(provider, LocalEmbedProvider)
        assert provider.dim == 768

        # Actually embed something
        vec = await provider.embed_query("test query")
        assert len(vec) == 768
