<div align="center">

# Universal Context

**Your AI agents forget everything. Universal Context fixes that.**

Operational memory for AI coding agents. Captures every session, distills working memory,<br>and gives your agents searchable context across your entire coding history.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-341%20passing-brightgreen.svg)]()
[![SurrealDB](https://img.shields.io/badge/SurrealDB-v3-6600FF.svg?logo=surrealdb&logoColor=white)](https://surrealdb.com)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa.svg)](https://github.com/sponsors/mysticaltech)

<br>

```
             ┌─────────────────────────────────────────┐
             │            UC Daemon                     │
             │                                         │
             │   Watcher              Worker           │
             │   ┌──────────────┐    ┌──────────────┐  │
             │   │ Claude Code  │    │ Summarizer   │  │
             │   │ Codex CLI    ├───►│ Embeddings   │  │
             │   │ Gemini CLI   │    │ Memory       │  │
             │   └──────┬───────┘    └──────┬───────┘  │
             │          └──────┬────────────┘          │
             │                 ▼                        │
             │          ┌────────────┐                  │
             │          │ SurrealDB  │                  │
             │          │ Graph DB   │                  │
             │          └────────────┘                  │
             └─────────────────────────────────────────┘
```

</div>

---

## The Problem

AI coding agents are stateless. Every session starts from zero. That architectural decision you made last week, the bug you spent an hour debugging, the pattern you established across your codebase — all gone. Your agents keep re-discovering the same things, making the same mistakes, asking the same questions.

## The Solution

Universal Context runs a background daemon that watches your AI coding sessions, captures every turn into a SurrealDB graph database with full provenance, summarizes them via LLM, generates local embeddings, and distills working memory per project. Your agents get searchable, scoped context — keyword, semantic, or hybrid — plus LLM-powered Q&A and agentic multi-hop reasoning across your entire history.

**Local-first.** Your data stays on your machine. No cloud, no telemetry, no third-party storage.

---

## Supported Tools

<table>
<tr>
<td align="center" width="33%">
<br>
<strong>Claude Code</strong><br>
<sub>Full JSONL conversation turns<br>Tool calls, file edits, reasoning</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<strong>Codex CLI</strong><br>
<sub>Event-stream responses<br>Code generation, shell commands</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<strong>Gemini CLI</strong><br>
<sub>JSON array conversations<br>Multi-turn context</sub>
<br><br>
</td>
</tr>
</table>

> More adapters welcome — [send a PR](#contributing).

---

## Features

<table>
<tr>
<td width="50%" valign="top">

### Search & Retrieval
- **BM25 full-text search** — keyword search with relevance ranking
- **Semantic vector search** — local embeddings, no API keys needed
- **Hybrid search** — RRF fusion of keyword + semantic
- **Scope filtering** — search within a project or across everything
- **Branch filtering** — scope queries to a specific git branch

### Working Memory
- **Auto-distilled** — LLM summarizes accumulated project knowledge
- **Injectable** — pipe memory into `CLAUDE.md`, `AGENTS.md`, etc.
- **Refreshable** — regenerate on demand as your project evolves

### Context & Q&A
- **Natural language Q&A** — "How does the auth middleware work?"
- **Agentic reasoning** — DSPy RLM multi-hop exploration of your session history
- **Context retrieval** — relevant sessions for a task description
- **JSON output** — every command supports `--json`

</td>
<td width="50%" valign="top">

### Git-Aware Intelligence
- **Scope identity** — worktrees and clones share one knowledge scope
- **Branch tracking** — every run records its branch + commit SHA
- **Merge detection** — detects when feature branches land on main
- **Cross-machine sharing** — export/import bundles with scope metadata

### Data Integrity
- **Graph provenance** — full `RELATE` edge chain from scope to summary
- **Secret redaction** — API keys, tokens, passwords stripped on capture
- **Immutable artifacts** — summaries are derived, never mutated

### Local-First by Default
- **Zero API keys required** — embedded DB + local ONNX embeddings
- **Optional LLM** — add an API key for better summaries and Q&A
- **Optional DB server** — upgrade to BM25 + HNSW when ready
- **TUI dashboard** — interactive overview, timeline, and search

</td>
</tr>
</table>

---

## Setup — Give This to Your Agent

> **The recommended way to set up UC is to paste this README into Claude Code, Codex, or any AI coding agent and tell it: "Install Universal Context for me."**
>
> Your agent can handle the entire setup autonomously.

If you prefer doing it yourself:

### Prerequisites

- **Python 3.13+** — `brew install python@3.13` or [python.org](https://python.org)
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh` ([docs](https://docs.astral.sh/uv/))

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

---

## SurrealDB Setup

> **Important: UC requires SurrealDB v3 (beta).** The default installer gets v2 — you must pass `--version` to get v3. UC will not work with SurrealDB v2.

UC works in two modes:

| Mode | Search Capabilities | Setup |
|------|-------------------|-------|
| **Embedded** (`file://`) | Substring match + brute-force cosine | Zero config — works out of the box |
| **Server** (`ws://`) | BM25 full-text + HNSW vector KNN | 2 minutes — full-power search |

The embedded mode works immediately with no extra setup. When you're ready for full-power search:

```bash
# Install SurrealDB v3 (must specify version — default installs v2)
curl -sSf https://install.surrealdb.com | sh -s -- --version v3.0.0-beta.4

# Verify you have v3
surreal version
# Expected: 3.0.0-beta.4

# Start the server
surreal start --bind 127.0.0.1:8000 --user root --pass root surrealkv://~/.surrealdb/data
```

Then point UC at it in `~/.uc/config.yaml`:

```yaml
db_url: "ws://127.0.0.1:8000"
db_user: "root"
db_pass: "root"
```

**What you get:** BM25 full-text search with relevance ranking, HNSW approximate nearest neighbor search for semantic queries, and automatic index rebuilds. Without it, search still works via substring matching and brute-force cosine similarity — just slower at scale.

---

## Configuration

`~/.uc/config.yaml`:

```yaml
# LLM for summarization and Q&A
# Works without any API key (extractive fallback), but much better with one
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
auto_detect_gemini: true
```

### LLM Providers

| Provider | Env Var | Notes |
|----------|---------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | **Recommended.** Multi-model gateway — one key, any model |
| `claude` | `ANTHROPIC_API_KEY` | Direct Anthropic API |
| `openai` | `OPENAI_API_KEY` | Direct OpenAI API |
| `auto` | Any of the above | Tries providers in priority order |

> **Recommended model:** Set `llm_model: "xai/grok-4.1-fast"` via OpenRouter. Fast, cheap, excellent at summarization. The default (Claude Haiku 4.5) works great too.

### Embedding Providers

| Provider | Model | Dims | API Key Required |
|----------|-------|------|:---:|
| `local` | EmbeddingGemma-300M (ONNX) | 768 | No |
| `openai` | text-embedding-3-small | 1536 | Yes |

The local model downloads ~600MB on first use and caches at `~/.cache/uc-models/`. No API keys, no PyTorch — pure ONNX inference.

---

## CLI Reference

<details>
<summary><strong>Search & Context</strong></summary>

```bash
uc search "query"                          # Keyword search (BM25)
uc search "query" --semantic               # Semantic vector search
uc search "query" --hybrid                 # Hybrid (keyword + semantic)
uc search "query" --project .              # Scoped to current project
uc context --json --context "task desc"    # Get relevant context for a task
uc context --project . --branch main       # Branch-filtered context
uc ask "question" --project .              # LLM-powered Q&A
uc reason "question" --project .          # Agentic multi-hop reasoning (DSPy RLM)
uc reason "question" --project . -v       # Show REPL trajectory
```

</details>

<details>
<summary><strong>Working Memory</strong></summary>

```bash
uc memory show --project .                 # View distilled project memory
uc memory refresh --project .              # Regenerate from latest sessions
uc memory inject --project .               # Write into AGENTS.md
uc memory inject -t CLAUDE.md              # Write into CLAUDE.md
uc memory eject --project .                # Remove injected memory
uc memory history --project .              # View memory versions
```

</details>

<details>
<summary><strong>Daemon & Status</strong></summary>

```bash
uc daemon start                            # Start in background
uc daemon start -f                         # Start in foreground (see logs)
uc daemon stop                             # Stop the daemon
uc status                                  # System overview
uc doctor                                  # Health check
```

</details>

<details>
<summary><strong>Inspect & Timeline</strong></summary>

```bash
uc timeline                                # Turns in latest run
uc timeline --branch main                  # Filter by branch
uc inspect turn:abc123                     # Full provenance chain for a turn
uc dashboard                               # Interactive TUI
```

</details>

<details>
<summary><strong>Sharing & Scopes</strong></summary>

```bash
uc share export run:abc123 -o bundle.json  # Export a run as v2 bundle
uc share import bundle.json --project .    # Import into current project scope
uc scope list --json                       # List all scopes
uc scope show <ref>                        # Show scope details
uc scope cleanup --dry-run                 # Find orphaned scopes
uc rebuild-index                           # Rebuild HNSW vector index
```

</details>

> Every command supports `--json` for machine-readable output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          UC Daemon                              │
│                                                                 │
│  ┌─────────────────────────┐    ┌───────────────────────────┐   │
│  │        Watcher          │    │         Worker            │   │
│  │                         │    │                           │   │
│  │  Adapters               │    │  TurnSummarizer           │   │
│  │   Claude (.jsonl)       │    │   LLM summarize           │   │
│  │   Codex  (events)       │    │   Embed (ONNX/API)        │   │
│  │   Gemini (.json)        │    │   Scope propagation       │   │
│  │         │               │    │                           │   │
│  │  Triggers               │    │  MemoryProcessor          │   │
│  │   Turn detection ───────┼────┼─► LLM distillation        │   │
│  │   Turn ingestion        │jobs│   Project memory          │   │
│  │                         │    │                           │   │
│  │  Git awareness          │    │  HNSW Rebuilder           │   │
│  │   Branch, commit SHA    │    │   Auto every 25 embeds    │   │
│  │   Canonical ID, merges  │    │                           │   │
│  └─────────────────────────┘    └───────────────────────────┘   │
│                                                                 │
│                       ┌─────────────┐                           │
│                       │  SurrealDB  │                           │
│                       │  Graph DB   │                           │
│                       └─────────────┘                           │
└─────────────────────────────────────────────────────────────────┘

Provenance: scope ← run ← turn ← transcript ← summary  (all RELATE edges)
Search:     BM25 full-text │ HNSW vector KNN │ brute-force cosine fallback
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-process daemon | RocksDB fcntl lock; watcher + worker as async task groups |
| Graph provenance | `RELATE` edges link every artifact to its source, fully traversable |
| Immutable summaries | Derived artifacts linked via `depends_on` edges, never mutated |
| Git-aware scopes | canonical_id from git remote URL — worktrees and clones share knowledge |
| Self-healing search | HNSW KNN with automatic brute-force cosine fallback |
| Denormalized scope | Scope field on artifacts avoids graph traversal on search hot path |

For the full architecture guide, see [AGENTS.md](AGENTS.md).

---

## Contributing

**PRs, not issues.** We don't use GitHub Issues for bug reports. If you found a bug, fix it and send a PR. If you want to add a feature, send a PR.

Every PR must pass **triple verification**:

| Check | Command |
|-------|---------|
| Tests pass | `pytest -x -v` (341+ tests) |
| Lint clean | `ruff check . && ruff format --check .` |
| Manual test | Verify your change works end-to-end |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

<div align="center">

## Support the Project

If Universal Context saves you time, consider sponsoring to keep it maintained and evolving.

<br>

<a href="https://github.com/sponsors/mysticaltech">
  <img src="https://img.shields.io/badge/Sponsor_on_GitHub-%E2%9D%A4-ea4aaa?style=for-the-badge&logo=github" alt="Sponsor on GitHub">
</a>

<br><br>

[Apache 2.0](LICENSE)

</div>
