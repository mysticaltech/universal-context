# Universal Context

**Your AI agents forget everything. Universal Context fixes that.**

Operational memory for AI coding agents. Captures every session, distills working memory, and gives your agents searchable context across your entire coding history.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-283%20passing-brightgreen.svg)]()
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa.svg)](https://github.com/sponsors/mysticaltech)

---

## The Problem

AI coding agents are stateless. Every session starts from zero. That architectural decision you made last week, the bug you spent an hour debugging, the pattern you established across your codebase — all gone. Your agents keep re-discovering the same things, making the same mistakes, asking the same questions.

## The Solution

Universal Context runs a background daemon that watches your AI coding sessions across **Claude Code**, **Codex CLI**, and **Gemini CLI**. It captures every turn into a SurrealDB graph database with full provenance, summarizes them via LLM, generates local embeddings, and distills working memory per project. Your agents get searchable, scoped context — keyword, semantic, or hybrid — plus LLM-powered Q&A across your entire history.

```
                         UC Daemon
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  Watcher                    Worker           │
  │  ┌────────────────┐        ┌──────────────┐  │
  │  │ Claude adapter │  jobs  │ Summarizer   │  │
  │  │ Codex adapter  ├───────►│ (LLM+embed)  │  │
  │  │ Gemini adapter │        │ Memory       │  │
  │  └───────┬────────┘        └──────┬───────┘  │
  │          │     SurrealDB          │          │
  │          └────────┬───────────────┘          │
  │                   │                          │
  └───────────────────┼──────────────────────────┘
                      │
              ┌───────▼────────┐
              │   Graph DB     │
              │ file:// or ws: │
              └────────────────┘
```

**Local-first.** Your data stays on your machine. No cloud, no telemetry, no third-party storage.

## Supported Tools

| Tool | Status | What UC Captures |
|------|--------|-----------------|
| **Claude Code** | Supported | Full JSONL conversation turns, tool calls, file edits |
| **Codex CLI** | Supported | Event-stream responses, code generation, shell commands |
| **Gemini CLI** | Supported | JSON array conversations, multi-turn context |

More adapters welcome — [send a PR](#contributing).

## Features

### Search & Retrieval
- **BM25 full-text search** — keyword search with relevance ranking
- **Semantic vector search** — find conceptually related sessions using local embeddings (EmbeddingGemma-300M, no API keys)
- **Hybrid search** — RRF fusion of keyword + semantic results
- **Scope filtering** — search within a single project or across everything
- **Branch filtering** — scope queries to a specific git branch

### Working Memory
- **Auto-distilled project memory** — LLM summarizes your project's accumulated knowledge into a concise brief
- **Injectable** — pipe working memory directly into agent config files (`CLAUDE.md`, `AGENTS.md`)
- **Refreshable** — regenerate memory on demand as your project evolves

### Context & Q&A
- **LLM-powered Q&A** — ask natural language questions about any project ("How does the auth middleware work?")
- **Context retrieval** — get relevant past sessions for a task description, formatted for agent consumption
- **JSON output** — every command supports `--json` for machine-readable output

### Git-Aware Intelligence
- **Scope identity** — worktrees and clones of the same repo automatically share one knowledge scope
- **Branch tracking** — every captured run records its branch and commit SHA
- **Merge detection** — UC detects when feature branch commits land on main
- **Cross-machine sharing** — export/import bundles with scope metadata for team knowledge transfer

### Data Integrity
- **Graph provenance** — every artifact linked via `RELATE` edges: summaries → transcripts → turns → runs → scopes
- **Secret redaction** — API keys, tokens, and passwords stripped before storage
- **Immutable artifacts** — summaries are derived, never mutated

### Local-First Operation
- **Zero API keys required** — embedded DB + local ONNX embeddings work out of the box
- **Optional LLM enhancement** — add OpenRouter, Claude, or OpenAI keys for better summaries and Q&A
- **Optional SurrealDB server** — upgrade to full BM25 + HNSW vector search when you want it

### Tools
- **TUI dashboard** — Textual-based interactive dashboard with Overview, Timeline, and Search tabs
- **CLI** — comprehensive command-line interface for all operations
- **Health checks** — `uc doctor` validates your setup and reports issues

## Setup — Give This to Your Agent

> **The recommended way to set up UC is to paste this section into Claude Code, Codex, or any AI coding agent and tell it: "Install Universal Context for me."**

Your agent can handle the entire setup. But if you prefer doing it yourself:

### Prerequisites

- **Python 3.13+** ([python.org](https://python.org) or `brew install python@3.13`)
- **uv** ([docs.astral.sh/uv](https://docs.astral.sh/uv/)) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Install

```bash
git clone https://github.com/mysticaltech/universal-context.git
cd universal-context
uv tool install --editable . --with anthropic
uc init
```

This puts `uc` on your PATH and creates `~/.uc/config.yaml`.

### Start the Daemon

```bash
uc daemon start -f
```

That's it. UC is now watching your AI coding sessions and building your operational memory.

## SurrealDB Setup

UC works in two modes:

| Mode | Search | Setup | Best For |
|------|--------|-------|----------|
| **Embedded** (`file://`) | Substring + brute-force cosine | Zero config | Getting started, small history |
| **Server** (`ws://`) | BM25 full-text + HNSW vector KNN | 2 minutes | Production, large history |

The embedded mode works immediately with no extra setup. When you're ready for full-power search:

```bash
# Install SurrealDB v3
curl -sSf https://install.surrealdb.com | sh

# Start the server (runs on port 8000)
surreal start --bind 127.0.0.1:8000 --user root --pass root surrealkv://~/.surrealdb/data
```

Then point UC at it in `~/.uc/config.yaml`:

```yaml
db_url: "ws://127.0.0.1:8000"
db_user: "root"
db_pass: "root"
```

**What you get with the server:** BM25 full-text search with relevance ranking, HNSW approximate nearest neighbor search for semantic queries, and automatic index rebuilds. Without it, search still works via substring matching and brute-force cosine similarity — just slower at scale.

## Configuration

`~/.uc/config.yaml`:

```yaml
# LLM for summarization and Q&A
# Extractive fallback used when no LLM key is configured
llm_provider: "openrouter"     # openrouter | claude | openai | auto
llm_model: ""                  # empty = provider default

# Embeddings — local by default, no API key needed
embed_provider: "local"        # local | openai

# API keys (or set via environment variables)
openrouter_api_key: ""         # or OPENROUTER_API_KEY env var
# anthropic_api_key: ""        # or ANTHROPIC_API_KEY env var
# openai_api_key: ""           # or OPENAI_API_KEY env var

# SurrealDB — optional server for full-power search
db_url: "ws://127.0.0.1:8000"
db_user: "root"
db_pass: "root"

# Which AI CLIs to watch
auto_detect_claude: true
auto_detect_codex: true
auto_detect_gemini: false       # opt-in
```

### LLM Providers

| Provider | Model | Env Var | Notes |
|----------|-------|---------|-------|
| `openrouter` | Provider default | `OPENROUTER_API_KEY` | Multi-model gateway, recommended |
| `claude` | Claude (Anthropic) | `ANTHROPIC_API_KEY` | Direct Anthropic API |
| `openai` | GPT series | `OPENAI_API_KEY` | OpenAI API |
| `auto` | Best available | Any of the above | Tries providers in order |

### Embedding Providers

| Provider | Model | Dims | API Key | Notes |
|----------|-------|------|---------|-------|
| `local` | EmbeddingGemma-300M | 768 | None | ONNX Runtime, ~600MB download on first use |
| `openai` | text-embedding-3-small | 1536 | `OPENAI_API_KEY` | Higher quality, requires API key |

## CLI Reference

### Search & Context

```bash
uc search "query"                          # Keyword search (BM25)
uc search "query" --semantic               # Semantic vector search
uc search "query" --hybrid                 # Hybrid (keyword + semantic)
uc search "query" --project .              # Scoped to current project
uc context --json --context "task desc"    # Get relevant context for a task
uc context --project . --branch main       # Branch-filtered context
uc ask "question" --project .              # LLM-powered Q&A
```

### Working Memory

```bash
uc memory show --project .                 # View distilled project memory
uc memory refresh --project .              # Regenerate from latest sessions
uc memory inject --project .               # Write into AGENTS.md
uc memory inject -t CLAUDE.md              # Write into CLAUDE.md
uc memory eject --project .                # Remove injected memory
uc memory history --project .              # View memory versions
```

### Daemon

```bash
uc daemon start                            # Start in background
uc daemon start -f                         # Start in foreground (see logs)
uc daemon stop                             # Stop the daemon
uc status                                  # System overview
uc doctor                                  # Health check
```

### Inspect & Timeline

```bash
uc timeline                                # Turns in latest run
uc timeline --branch main                  # Filter by branch
uc inspect turn:abc123                     # Full provenance chain for a turn
uc dashboard                               # Interactive TUI
```

### Sharing & Scopes

```bash
uc share export run:abc123 -o bundle.json  # Export a run as v2 bundle
uc share import bundle.json --project .    # Import into current project scope
uc scope list --json                       # List all scopes
uc scope show <ref>                        # Show scope details
uc scope cleanup --dry-run                 # Find orphaned scopes
uc rebuild-index                           # Rebuild HNSW vector index
```

Every command supports `--json` for machine-readable output.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          UC Daemon                              │
│                                                                 │
│  ┌─────────────────────────┐    ┌───────────────────────────┐   │
│  │        Watcher          │    │         Worker            │   │
│  │                         │    │                           │   │
│  │  ┌───────────────────┐  │    │  ┌─────────────────────┐  │   │
│  │  │ Adapters          │  │    │  │ TurnSummarizer      │  │   │
│  │  │  Claude (.jsonl)  │  │    │  │  LLM summarize      │  │   │
│  │  │  Codex  (events)  │  │    │  │  Embed (ONNX/API)   │  │   │
│  │  │  Gemini (.json)   │  │    │  │  Scope propagation  │  │   │
│  │  └────────┬──────────┘  │    │  └─────────────────────┘  │   │
│  │           │             │    │  ┌─────────────────────┐  │   │
│  │  ┌────────▼──────────┐  │    │  │ MemoryProcessor     │  │   │
│  │  │ Triggers          │  │    │  │  LLM distillation   │  │   │
│  │  │  Turn detection   │──┼────┼─►│  Project memory     │  │   │
│  │  │  Turn ingestion   │  │jobs│  └─────────────────────┘  │   │
│  │  └───────────────────┘  │    │  ┌─────────────────────┐  │   │
│  │                         │    │  │ HNSW Rebuilder       │  │   │
│  │  Git: branch, commit,   │    │  │  Auto every 25 new  │  │   │
│  │  canonical_id, merges   │    │  │  embeddings          │  │   │
│  └─────────────────────────┘    │  └─────────────────────┘  │   │
│                                 └───────────────────────────┘   │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │  SurrealDB  │                               │
│                   │  Graph DB   │                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

Storage: scope ← run ← turn ← transcript ← summary (all RELATE edges)
Search:  BM25 full-text │ HNSW vector KNN │ brute-force cosine fallback
```

### Key Design Decisions

- **Single-process daemon** — RocksDB fcntl lock prevents multi-process; watcher + worker run as async task groups
- **Graph provenance** — `RELATE` edges link every artifact to its source, fully traversable
- **Immutable summaries** — derived artifacts linked via `depends_on` edges, never mutated
- **Git-aware scopes** — canonical_id from git remote URL means worktrees and clones share knowledge
- **Self-healing search** — HNSW KNN with automatic brute-force cosine fallback when index is stale
- **Denormalized scope** — scope field on artifacts avoids graph traversal on the search hot path

For the full architecture guide, see [CLAUDE.md](CLAUDE.md).

## Contributing

**PRs, not issues.** We don't use GitHub Issues for bug reports. If you found a bug, fix it and send a PR. If you want to add a feature, send a PR.

Every PR must pass triple verification:

1. **Tests pass** — `pytest -x -v` (283+ tests)
2. **Lint clean** — `ruff check . && ruff format --check .`
3. **Manual test** — verify your change works end-to-end

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## Support the Project

If Universal Context saves you time, consider sponsoring to keep it maintained and evolving.

<a href="https://github.com/sponsors/mysticaltech">
  <img src="https://img.shields.io/badge/Sponsor_on_GitHub-%E2%9D%A4-ea4aaa?style=for-the-badge&logo=github" alt="Sponsor on GitHub">
</a>

## License

[Apache 2.0](LICENSE)
