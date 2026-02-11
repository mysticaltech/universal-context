"""SurrealDB schema definitions for Universal Context.

Progressive schema: start schemaless for flexibility, define fields
and indexes for performance-critical tables. Idempotent — safe to re-run.
"""

from __future__ import annotations

from .client import UCDatabase

SCHEMA_VERSION = 4

# Each statement is a separate list item to avoid semicolon-splitting issues.
SCHEMA_STATEMENTS: list[str] = [
    # ============================================================
    # CORE RECORD TABLES
    # ============================================================

    # Scope
    "DEFINE TABLE IF NOT EXISTS scope SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS name ON scope TYPE string",
    "DEFINE FIELD IF NOT EXISTS path ON scope TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS created_at ON scope TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS canonical_id ON scope TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS metadata ON scope TYPE object FLEXIBLE DEFAULT {}",

    # Run
    "DEFINE TABLE IF NOT EXISTS run SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS scope ON run TYPE record<scope>",
    "DEFINE FIELD IF NOT EXISTS agent_type ON run TYPE string",
    "DEFINE FIELD IF NOT EXISTS started_at ON run TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS ended_at ON run TYPE option<datetime>",
    'DEFINE FIELD IF NOT EXISTS status ON run TYPE string DEFAULT "active"',
    "DEFINE FIELD IF NOT EXISTS session_path ON run TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS branch ON run TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS commit_sha ON run TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS merged_to ON run TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS metadata ON run TYPE object FLEXIBLE DEFAULT {}",

    # Turn
    "DEFINE TABLE IF NOT EXISTS turn SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS run ON turn TYPE record<run>",
    "DEFINE FIELD IF NOT EXISTS sequence ON turn TYPE int",
    "DEFINE FIELD IF NOT EXISTS started_at ON turn TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS ended_at ON turn TYPE option<datetime>",
    "DEFINE FIELD IF NOT EXISTS user_message ON turn TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS metadata ON turn TYPE object FLEXIBLE DEFAULT {}",

    # Step
    "DEFINE TABLE IF NOT EXISTS step SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS turn ON step TYPE record<turn>",
    "DEFINE FIELD IF NOT EXISTS sequence ON step TYPE int",
    "DEFINE FIELD IF NOT EXISTS action_type ON step TYPE string",
    "DEFINE FIELD IF NOT EXISTS action_data ON step TYPE object FLEXIBLE DEFAULT {}",
    "DEFINE FIELD IF NOT EXISTS timestamp ON step TYPE datetime DEFAULT time::now()",

    # Artifact
    "DEFINE TABLE IF NOT EXISTS artifact SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS kind ON artifact TYPE string",
    "DEFINE FIELD IF NOT EXISTS content ON artifact TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS content_hash ON artifact TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS blob_path ON artifact TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS created_at ON artifact TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS metadata ON artifact TYPE object FLEXIBLE DEFAULT {}",
    "DEFINE FIELD IF NOT EXISTS embedding ON artifact TYPE option<array<float>>",
    "DEFINE FIELD IF NOT EXISTS scope ON artifact TYPE option<record<scope>>",

    # Checkpoint
    "DEFINE TABLE IF NOT EXISTS checkpoint SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS run ON checkpoint TYPE record<run>",
    "DEFINE FIELD IF NOT EXISTS turn ON checkpoint TYPE record<turn>",
    "DEFINE FIELD IF NOT EXISTS label ON checkpoint TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS created_at ON checkpoint TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS state ON checkpoint TYPE object FLEXIBLE DEFAULT {}",

    # Job
    "DEFINE TABLE IF NOT EXISTS job SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS job_type ON job TYPE string",
    'DEFINE FIELD IF NOT EXISTS status ON job TYPE string DEFAULT "pending"',
    "DEFINE FIELD IF NOT EXISTS target ON job TYPE string",
    "DEFINE FIELD IF NOT EXISTS priority ON job TYPE int DEFAULT 0",
    "DEFINE FIELD IF NOT EXISTS attempts ON job TYPE int DEFAULT 0",
    "DEFINE FIELD IF NOT EXISTS max_attempts ON job TYPE int DEFAULT 10",
    "DEFINE FIELD IF NOT EXISTS created_at ON job TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD IF NOT EXISTS started_at ON job TYPE option<datetime>",
    "DEFINE FIELD IF NOT EXISTS completed_at ON job TYPE option<datetime>",
    "DEFINE FIELD IF NOT EXISTS error ON job TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS result ON job TYPE option<object> FLEXIBLE",

    # ============================================================
    # GRAPH EDGE TABLES (schemaless — RELATE manages in/out)
    # ============================================================
    "DEFINE TABLE IF NOT EXISTS produced SCHEMALESS",
    "DEFINE TABLE IF NOT EXISTS contains SCHEMALESS",
    "DEFINE TABLE IF NOT EXISTS depends_on SCHEMALESS",
    "DEFINE TABLE IF NOT EXISTS checkpoint_at SCHEMALESS",

    # ============================================================
    # INDEXES
    # ============================================================
    "DEFINE INDEX IF NOT EXISTS idx_scope_path ON scope FIELDS path",
    "DEFINE INDEX IF NOT EXISTS idx_scope_canonical ON scope FIELDS canonical_id UNIQUE",
    "DEFINE INDEX IF NOT EXISTS idx_scope_name ON scope FIELDS name",
    "DEFINE INDEX IF NOT EXISTS idx_run_scope ON run FIELDS scope",
    "DEFINE INDEX IF NOT EXISTS idx_run_status ON run FIELDS status",
    "DEFINE INDEX IF NOT EXISTS idx_turn_run ON turn FIELDS run",
    "DEFINE INDEX IF NOT EXISTS idx_turn_sequence ON turn FIELDS run, sequence",
    "DEFINE INDEX IF NOT EXISTS idx_step_turn ON step FIELDS turn",
    "DEFINE INDEX IF NOT EXISTS idx_artifact_kind ON artifact FIELDS kind",
    "DEFINE INDEX IF NOT EXISTS idx_artifact_scope ON artifact FIELDS scope",
    "DEFINE INDEX IF NOT EXISTS idx_job_status ON job FIELDS status, priority",
    "DEFINE INDEX IF NOT EXISTS idx_job_type ON job FIELDS job_type",

]

# Server-only statements — require SurrealDB v3 server.
# The embedded SDK bundles a v2 engine that cannot parse FULLTEXT ANALYZER
# or reliably populate HNSW indexes on live inserts.
SERVER_STATEMENTS: list[str] = [
    # FTS (BM25) — v3 uses FULLTEXT ANALYZER (v2 used SEARCH ANALYZER)
    "DEFINE ANALYZER IF NOT EXISTS uc_text TOKENIZERS blank, class"
    " FILTERS lowercase, ascii, snowball(english)",
    "DEFINE INDEX IF NOT EXISTS idx_artifact_fts ON artifact"
    " FIELDS content FULLTEXT ANALYZER uc_text BM25",
]

# HNSW index template — dimension is injected at runtime from the embed provider
_HNSW_TEMPLATE = (
    "DEFINE INDEX IF NOT EXISTS idx_artifact_embedding ON artifact"
    " FIELDS embedding HNSW DIMENSION {dim} DIST COSINE TYPE F32"
)

# Schema version marker (separate — uses UPSERT pattern)
META_STATEMENT = 'UPSERT meta:schema SET version = 4, updated_at = time::now()'


async def rebuild_hnsw_index(db: UCDatabase, embedding_dim: int = 768) -> bool:
    """Drop and recreate the HNSW index to incorporate new embeddings.

    HNSW indexes in SurrealDB v3 are build-once — data inserted after
    DEFINE INDEX is invisible to KNN queries. This forces a full rebuild.

    Returns True if rebuilt (server mode), False if skipped (embedded).
    """
    if not db.is_server:
        return False
    await db.query("REMOVE INDEX IF EXISTS idx_artifact_embedding ON artifact")
    await db.query(_HNSW_TEMPLATE.format(dim=embedding_dim))
    return True


async def apply_schema(db: UCDatabase, embedding_dim: int = 768) -> None:
    """Apply the full schema to the database. Idempotent.

    Args:
        db: Database connection.
        embedding_dim: HNSW index dimension — auto-detected from the embed provider.
                       Default 768 = EmbeddingGemma-300M.
    """
    # v2 → v3 migration: drop old UNIQUE path index before redefining as non-unique
    await db.query("REMOVE INDEX IF EXISTS idx_scope_path ON scope")

    for statement in SCHEMA_STATEMENTS:
        await db.query(statement)
    if db.is_server:
        for statement in SERVER_STATEMENTS:
            await db.query(statement)
        try:
            await db.query(_HNSW_TEMPLATE.format(dim=embedding_dim))
        except RuntimeError:
            # HNSW index creation can fail on SurrealDB v3 beta (corrupted
            # index data, missing files, etc.).  Search falls back to
            # brute-force cosine automatically, so this is non-fatal.
            pass
    await db.query(META_STATEMENT)
