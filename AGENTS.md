# Universal Context — Agent Development Guide

## What This Is

Operational memory engine for AI agents. Captures sessions from Claude Code, Codex CLI, and Gemini CLI into a SurrealDB graph database with full provenance. Supports BM25, semantic vector search, hybrid search, working memory distillation, LLM-powered Q&A, and agentic multi-hop reasoning via DSPy RLM.

> Alpha reset policy (2026-02-12): CLI surface is intentionally breaking while UX is simplified. No backward compatibility guarantees yet.

## Quick Reference

```bash
# Dev install
uv pip install -e ".[dev,llm]"

# CLI
uc --help
uc doctor
uc status
uc "question about project memory"
uc find "query" --project .
uc ask "question" --project .
uc ask "question" --project . --deep
uc daemon start -f
uc memory sync --project .

# Working memory
uc memory show --project .
uc memory refresh --project .
uc memory inject --project .

# Admin-only commands
uc admin --help
uc admin search "query"
uc admin reason "question" --verbose
uc admin context --project . --branch main --json
uc admin share export run:abc123 -o bundle.json
uc admin share import bundle.json --project .
uc admin scope list --json
uc admin rebuild-index
uc admin timeline --branch main
uc admin inspect turn:abc123

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
- **Git-aware scopes**: canonical_id from git remote URL → worktrees/clones share one scope, branch + commit_sha + merged_to on runs
- **Automatic scope backfill**: daemon startup backfills canonical_id on legacy scopes, merging duplicates
- **Merge detection**: watcher detects when feature branch commits are ancestors of current branch via `git merge-base --is-ancestor`
- **Cross-machine sharing**: v2 bundles carry scope metadata; import matches by canonical_id or explicit `--project`
- **Scope-filtered search**: denormalized `scope` field on artifacts for O(1) scope filtering
- **Self-healing vector search**: HNSW KNN with automatic brute-force cosine fallback

## Project Structure

```
universal_context/
├── cli.py                  # Typer CLI (public surface + `admin` operator namespace)
├── config.py               # UCConfig dataclass, YAML load/save
├── embed.py                # EmbedProvider ABC + Local (EmbeddingGemma/ONNX) + OpenAI
├── git.py                  # Git-aware scope identity (canonical_id, branch, commit_sha, is_ancestor)
├── llm.py                  # LLM provider factory (Claude, OpenAI, OpenRouter)
├── reason.py               # DSPy RLM integration (agentic reasoning over session history)
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
├── sharing/                # Export/import v2 bundles (scope-aware) + checkpoints
├── tui/                    # Textual dashboard (Overview, Timeline, Search tabs)
└── models/types.py         # Pydantic domain models + StrEnum types

tests/                      # 341 tests
├── test_git.py             # Git URL normalization, canonical_id, branch, cross-worktree, is_ancestor
├── test_reason.py          # RLM integration: AsyncBridge, LocalInterpreter, tools, LM config
├── test_db.py              # Schema, queries, search
├── test_search.py          # Scope filtering, embedded fallbacks, RRF
├── test_embed.py           # Embedding providers + vector search
├── test_adapters.py        # Adapters + triggers
├── test_daemon.py          # Watcher, worker, summarizer, backfill, merge detection
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
12. **D12**: Git-aware scope identity — canonical_id = normalized git remote URL > git-common-dir > path://. Worktrees and clones of the same repo share one scope. Branch + commit_sha captured on each run for provenance
13. **D13**: Automatic backfill — daemon startup runs `backfill_canonical_ids()` to migrate legacy scopes, merging duplicates
14. **D14**: Merge detection — watcher tags runs from feature branches whose commits are ancestors of the current branch (`merged_to` field)
15. **D15**: v2 share bundles — carry scope metadata (name, path, canonical_id) for cross-machine scope matching on import
16. **D16**: SurrealDB v2 embedded NONE quirk — `WHERE field = NONE` doesn't match; use `WHERE !field` for falsy checks on option fields
17. **D17**: DSPy RLM for agentic reasoning — LLM writes Python in a REPL loop to explore the graph database. Core dependency (requires DSPy ≥3.1.1 where RLM was introduced). LocalInterpreter (in-process, no Deno/WASM), AsyncBridge wraps main event loop, tools submit coroutines via `run_coroutine_threadsafe()`
18. **D18**: HNSW schema resilience — `apply_schema()` catches HNSW index creation failures gracefully (SurrealDB v3 beta corruption). Search self-heals via brute-force cosine fallback

## SurrealDB Notes

- **v3 only** — targets SurrealDB v3.0.0-beta.4+
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
- **Mitigations**: `apply_schema()` catches HNSW creation failures (non-fatal). `semantic_search()` auto-falls back to brute-force cosine when HNSW returns 0 results. Worker auto-rebuilds HNSW after 25 new embeddings. `uc admin rebuild-index` available for manual rebuild.
- **Data corruption**: HNSW index files can go missing on disk (`IO error: No such file or directory`). `REMOVE INDEX` succeeds but `DEFINE INDEX` fails until server restart with fresh data path. Tracked upstream.

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
