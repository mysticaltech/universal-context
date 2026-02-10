"""Tests for scope-filtered search, embedded fallbacks, RRF merge, and backfill."""

from __future__ import annotations

import hashlib

import pytest

from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    _merge_rrf,
    backfill_artifact_scopes,
    create_artifact,
    create_run,
    create_scope,
    hybrid_search,
    search_artifacts,
    semantic_search,
    store_embedding,
)
from universal_context.db.schema import apply_schema


@pytest.fixture
async def db(tmp_path):
    d = UCDatabase(f"mem://test_search_{id(tmp_path)}")
    await d.connect()
    await apply_schema(d)
    yield d
    await d.close()


def _fake_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    h = hashlib.sha256(text.encode()).digest()
    values = []
    for i in range(dim):
        byte_idx = i % len(h)
        values.append((h[byte_idx] + i) % 256 / 255.0)
    return values


async def _setup_two_scopes(db: UCDatabase):
    """Create two scopes with artifacts — one for each project."""
    scope_a = await create_scope(db, "project-alpha", "/home/user/alpha")
    scope_b = await create_scope(db, "project-beta", "/home/user/beta")
    sid_a = str(scope_a["id"])
    sid_b = str(scope_b["id"])

    # Create artifacts with explicit scope
    aid_a1 = await create_artifact(
        db, "summary", content="Alpha uses JWT authentication middleware",
        scope_id=sid_a,
    )
    aid_a2 = await create_artifact(
        db, "summary", content="Alpha database migration system refactored",
        scope_id=sid_a,
    )
    aid_b1 = await create_artifact(
        db, "summary", content="Beta uses OAuth2 authentication flow",
        scope_id=sid_b,
    )
    aid_b2 = await create_artifact(
        db, "summary", content="Beta frontend uses React with TypeScript",
        scope_id=sid_b,
    )

    return {
        "scope_a": sid_a, "scope_b": sid_b,
        "aids_a": [aid_a1, aid_a2], "aids_b": [aid_b1, aid_b2],
    }


# ============================================================
# SCOPE-FILTERED SEARCH
# ============================================================


class TestScopeFilteredSearch:
    @pytest.mark.asyncio
    async def test_search_with_scope_returns_only_matching(self, db):
        """Search with scope_id only returns artifacts from that scope."""
        info = await _setup_two_scopes(db)

        # Search for "authentication" scoped to alpha
        results = await search_artifacts(
            db, "authentication", scope_id=info["scope_a"],
        )
        assert len(results) == 1
        assert "JWT" in results[0]["content"]

        # Search for "authentication" scoped to beta
        results = await search_artifacts(
            db, "authentication", scope_id=info["scope_b"],
        )
        assert len(results) == 1
        assert "OAuth2" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_without_scope_returns_all(self, db):
        """Search without scope_id returns results from all scopes."""
        await _setup_two_scopes(db)

        results = await search_artifacts(db, "authentication")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_scope_no_matches(self, db):
        """Search with scope_id returns empty when no matches in that scope."""
        info = await _setup_two_scopes(db)

        # "React" only exists in beta
        results = await search_artifacts(
            db, "React", scope_id=info["scope_a"],
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_kind_filter_with_scope(self, db):
        """Kind filter and scope filter work together."""
        info = await _setup_two_scopes(db)

        # Add a transcript artifact to alpha
        await create_artifact(
            db, "transcript", content="raw authentication transcript",
            scope_id=info["scope_a"],
        )

        results = await search_artifacts(
            db, "authentication", kind="summary", scope_id=info["scope_a"],
        )
        assert len(results) == 1
        assert results[0]["kind"] == "summary"


# ============================================================
# EMBEDDED FALLBACK SEARCH
# ============================================================


class TestEmbeddedFallbackSearch:
    @pytest.mark.asyncio
    async def test_text_search_returns_results_on_embedded(self, db):
        """Text search works on embedded via substring match."""
        await create_artifact(db, "summary", content="debugging the auth module")
        await create_artifact(db, "summary", content="fixing CSS styles")

        results = await search_artifacts(db, "auth")
        assert len(results) == 1
        assert "auth" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_text_search_case_insensitive(self, db):
        """Embedded text search is case-insensitive."""
        await create_artifact(db, "summary", content="Fixed Authentication Bug")

        results = await search_artifacts(db, "authentication")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_vector_search_returns_results_on_embedded(self, db):
        """Vector search works on embedded via brute-force cosine."""
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
        assert len(results) > 0
        # Results should have a score field from cosine similarity
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_vector_search_scope_filtered(self, db):
        """Vector search with scope filter on embedded."""
        info = await _setup_two_scopes(db)

        # Add embeddings to all artifacts
        for aid in info["aids_a"] + info["aids_b"]:
            result = await db.query(f"SELECT content FROM {aid}")
            content = result[0]["content"]
            await store_embedding(db, aid, _fake_embedding(content))

        query_emb = _fake_embedding("authentication")
        results = await semantic_search(
            db, query_emb, scope_id=info["scope_a"], limit=5,
        )
        # All results should be from scope_a
        for r in results:
            assert str(r.get("scope")) == info["scope_a"]

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results_on_embedded(self, db):
        """Hybrid search works on embedded via Python RRF merge."""
        contents = [
            "JWT authentication middleware added",
            "Database migration scripts updated",
        ]
        for c in contents:
            aid = await create_artifact(db, kind="summary", content=c)
            await store_embedding(db, aid, _fake_embedding(c))

        query_emb = _fake_embedding("JWT auth")
        results = await hybrid_search(
            db, "JWT", query_emb, kind="summary", limit=5,
        )
        assert len(results) > 0


# ============================================================
# RRF MERGE
# ============================================================


class TestRRFMerge:
    def test_empty_inputs(self):
        assert _merge_rrf([], [], limit=10) == []

    def test_single_source(self):
        docs = [{"id": "artifact:a", "content": "hello"}]
        result = _merge_rrf(docs, [], limit=10)
        assert len(result) == 1
        assert result[0]["id"] == "artifact:a"
        assert "rrf_score" in result[0]

    def test_overlapping_results_ranked_higher(self):
        """Documents appearing in both lists get higher RRF scores."""
        text = [
            {"id": "artifact:a", "content": "both"},
            {"id": "artifact:b", "content": "text-only"},
        ]
        vector = [
            {"id": "artifact:a", "content": "both"},
            {"id": "artifact:c", "content": "vector-only"},
        ]
        result = _merge_rrf(text, vector, limit=10)
        assert result[0]["id"] == "artifact:a"  # Appears in both → highest

    def test_limit_respected(self):
        text = [{"id": f"artifact:{i}", "content": f"t{i}"} for i in range(5)]
        vector = [{"id": f"artifact:{i+5}", "content": f"v{i}"} for i in range(5)]
        result = _merge_rrf(text, vector, limit=3)
        assert len(result) == 3

    def test_rrf_score_decreases_with_rank(self):
        """Higher ranked documents get higher RRF scores."""
        text = [
            {"id": "artifact:a", "content": "first"},
            {"id": "artifact:b", "content": "second"},
        ]
        result = _merge_rrf(text, [], limit=10)
        assert result[0]["rrf_score"] > result[1]["rrf_score"]


# ============================================================
# BACKFILL
# ============================================================


class TestBackfillScopes:
    @pytest.mark.asyncio
    async def test_backfill_sets_scope_on_turn_artifacts(self, db):
        """Backfill resolves scope via turn->run->scope provenance."""
        scope = await create_scope(db, "backfill-test", "/tmp/backfill")
        scope_id = str(scope["id"])
        run = await create_run(db, scope_id, "claude")
        run_id = str(run["id"])

        # create_turn_with_artifact now sets scope automatically,
        # so we test backfill by manually creating an artifact WITHOUT scope
        aid = await create_artifact(db, "transcript", content="test content")
        # Manually create the produced edge
        turn_result = await db.query(
            f"CREATE turn:{hashlib.sha256(b'test').hexdigest()[:12]} "
            f"SET run = {run_id}, sequence = 1"
        )
        turn_id = str(turn_result[0]["id"])
        await db.query(f"RELATE {run_id}->contains->{turn_id}")
        await db.query(f"RELATE {turn_id}->produced->{aid}")

        # Verify no scope set
        result = await db.query(f"SELECT scope FROM {aid}")
        assert result[0].get("scope") is None

        # Run backfill
        count = await backfill_artifact_scopes(db)
        assert count >= 1

        # Verify scope now set
        result = await db.query(f"SELECT scope FROM {aid}")
        assert str(result[0]["scope"]) == scope_id

    @pytest.mark.asyncio
    async def test_backfill_working_memory_via_metadata(self, db):
        """Backfill sets scope on working_memory artifacts using metadata.scope_id."""
        scope = await create_scope(db, "wm-backfill", "/tmp/wm")
        scope_id = str(scope["id"])

        # Create a working_memory artifact WITHOUT scope field
        aid = await create_artifact(
            db, "working_memory", content="project state",
            metadata={"scope_id": scope_id, "method": "llm"},
        )

        count = await backfill_artifact_scopes(db)
        assert count >= 1

        result = await db.query(f"SELECT scope FROM {aid}")
        assert str(result[0]["scope"]) == scope_id

    @pytest.mark.asyncio
    async def test_backfill_idempotent(self, db):
        """Running backfill twice doesn't double-count."""
        scope = await create_scope(db, "idem-test", "/tmp/idem")
        scope_id = str(scope["id"])

        await create_artifact(
            db, "working_memory", content="test",
            metadata={"scope_id": scope_id, "method": "llm"},
        )

        count1 = await backfill_artifact_scopes(db)
        count2 = await backfill_artifact_scopes(db)
        assert count1 >= 1
        assert count2 == 0  # Already backfilled
