# Universal Context — Agent Development Guide

## What This Is

Operational memory engine for AI agents. Captures sessions from Claude Code, Codex CLI, and Gemini CLI into a SurrealDB graph database with full provenance. Supports BM25, semantic vector search, hybrid search, working memory distillation, and LLM-powered Q&A.

## Quick Reference

```bash
# Dev install
uv pip install -e ".[dev,llm]"

# CLI
uc --help
uc init
uc doctor
uc status
uc search "query" --project .
uc ask "question" --project .
uc context --json --context "task description"
uc rebuild-index
uc timeline
uc inspect turn:abc123
uc daemon start -f
uc dashboard

# Working memory
uc memory show --project .
uc memory refresh --project .
uc memory inject --project .

# Test & lint
pytest
pytest -x -v
ruff check .
ruff format .
```

## Architecture

- **Single daemon process**: watcher + worker as async task groups (RocksDB fcntl lock = single-process)
- **SurrealDB graph DB**: `file://` embedded or `ws://` server, graph edges for provenance
- **Adapter/Trigger pattern**: pluggable per-runtime session detection (Claude, Codex, Gemini)
- **Summaries are artifacts**: derived, immutable, linked via `depends_on` provenance edges
- **Scope-filtered search**: denormalized `scope` field on artifacts for O(1) scope filtering
- **Self-healing vector search**: HNSW KNN with automatic brute-force cosine fallback

## Project Structure

```
universal_context/
├── cli.py                  # Typer CLI (search, ask, context, timeline, inspect, daemon, memory)
├── config.py               # UCConfig dataclass, YAML load/save
├── embed.py                # EmbedProvider ABC + Local (EmbeddingGemma/ONNX) + OpenAI
├── llm.py                  # LLM provider factory (Claude, OpenAI, OpenRouter)
├── redact.py               # Secret redaction (API keys, tokens, passwords)
├── db/
│   ├── client.py           # UCDatabase (async, embedded + server)
│   ├── schema.py           # SCHEMAFULL tables, indexes, FTS, HNSW
│   └── queries.py          # CRUD, search, provenance, job queue, backfill
├── daemon/
│   ├── core.py             # UCDaemon (TaskGroup: watcher + worker)
│   ├── watcher.py          # Session discovery + turn ingestion
│   ├── worker.py           # Job claim loop + auto HNSW rebuild
│   └── processors/
│       ├── summarizer.py   # TurnSummarizer (LLM + embed + scope propagation)
│       └── memory.py       # WorkingMemoryProcessor (LLM distillation)
├── adapters/               # Session discovery: claude.py, codex.py, gemini.py
├── triggers/               # Turn detection: claude_trigger, codex_trigger, gemini_trigger
├── sharing/                # Export/import bundles + checkpoints
├── tui/                    # Textual dashboard (Overview, Timeline, Search tabs)
└── models/types.py         # Pydantic domain models + StrEnum types

tests/                      # 233 tests
├── test_db.py              # Schema, queries, search
├── test_search.py          # Scope filtering, embedded fallbacks, RRF
├── test_embed.py           # Embedding providers + vector search
├── test_adapters.py        # Adapters + triggers
├── test_daemon.py          # Watcher, worker, summarizer
├── test_cli_json.py        # CLI --json output
├── test_e2e.py             # Full pipeline integration
└── ...                     # models, config, sharing, redact, llm, memory
```

## Conventions

- Python 3.13+, type hints everywhere
- `from __future__ import annotations` in every file
- Pydantic v2 for data models, `StrEnum` for enumerations
- Async by default for DB and daemon code
- No silent failures — always surface errors clearly
- Secret redaction applied to all captured content

## Key Design Decisions

1. **D1**: Single-process daemon (RocksDB fcntl lock prevents multi-process)
2. **D2**: Graph edges for provenance (`RELATE` + arrow traversal)
3. **D3**: Summaries are immutable artifacts with `depends_on` edges
4. **D4**: Job claim loop (not change feeds — those are for audit/replay)
5. **D5**: SurrealQL transactions for atomic turn capture
6. **D6**: SCHEMAFULL tables with `DEFAULT {}` for object fields
7. **D7**: SurrealDB v3+ only, Python SDK v1.0.8 from main branch
8. **D8**: Config at `~/.uc/`, DB at `~/.uc/data/surreal.db/`
9. **D9**: Denormalized `scope` on artifacts (avoids graph traversal on search hot path)
10. **D10**: Embedded search fallbacks — substring match for text, brute-force cosine for vector, Python RRF merge for hybrid
11. **D11**: HNSW self-healing — `semantic_search()` tries KNN first, falls back to brute-force cosine if index is stale; worker auto-rebuilds HNSW after every 25 new embeddings

## SurrealDB Notes

- **v3 only** — targets SurrealDB v3.0.0-beta.3+
- Python SDK: `surrealdb @ git+https://github.com/surrealdb/surrealdb.py.git@1ff4470e`
- Edge tables must be SCHEMALESS (`RELATE` manages `in`/`out` automatically)
- Object fields on SCHEMAFULL tables need `DEFAULT {}` or SurrealDB rejects NONE values
- Graph traversal: `->edge->target` (forward), `<-edge<-source` (reverse)
- `BEGIN TRANSACTION; ...; COMMIT TRANSACTION` for atomic writes (client-side txn not available in embedded)
- `RecordID` objects aren't JSON-serializable — use `_sanitize_record()` in cli.py

### v3 Syntax

- BM25: `FULLTEXT ANALYZER` (not v2's `SEARCH ANALYZER`). `@@` operator unchanged.
- MTREE index: removed in v3 (parse error).
- KNN: must use two-param `<|k,ef|>` (e.g. `<|3,40|>`). Single-param `<|k|>` is broken.

### HNSW Index (v3 beta)

- **Build-once bug**: data inserted after `DEFINE INDEX ... HNSW` is invisible to KNN.
- **Workaround**: `REMOVE INDEX` + `DEFINE INDEX` forces rebuild. `REBUILD INDEX` is broken.
- **Mitigations**: `semantic_search()` auto-falls back to brute-force cosine when HNSW returns 0 results. Worker auto-rebuilds HNSW after 25 new embeddings. `uc rebuild-index` available for manual rebuild.

### Schema Split

- `SCHEMA_STATEMENTS`: core tables, fields, basic indexes — both embedded and server
- `SERVER_STATEMENTS`: FTS (`FULLTEXT ANALYZER`) and HNSW indexes — server only
- `apply_schema(db, embedding_dim=N)` checks `db.is_server`
- All search functions work on both modes with automatic fallbacks

### Embedded SDK Limitations

- Python SDK bundles a v2 engine internally
- `FULLTEXT ANALYZER` parse error on embedded (v2 uses `SEARCH ANALYZER`)
- HNSW KNN returns 0 on embedded (index never populates)
- `vector::similarity::cosine()` brute-force works on embedded
- Production targets v3 server. Embedded mode uses fallbacks.

## Embedding

- **Default**: `"local"` — EmbeddingGemma-300M via ONNX Runtime (768 dims, no API keys, no PyTorch)
- `"openai"` — text-embedding-3-small (1536 dims, requires `OPENAI_API_KEY`)
- `EmbedProvider` ABC: `dim`, `embed_query()` (search prefix), `embed_document()` (doc prefix)
- HNSW dimension is dynamic: `apply_schema(db, embedding_dim=provider.dim)`

## Testing

- Unit tests use `mem://` (embedded) — verify storage, schema, CRUD, provenance, search fallbacks
- Embedding tests use fake deterministic vectors + `MagicMock` providers
- Full search integration (BM25 + HNSW KNN) requires a running v3 server
- `pytest -m "not integration"` skips tests that need API keys
