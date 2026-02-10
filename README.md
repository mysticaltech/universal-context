# Universal Context

Operational memory for AI coding agents. Local-first, sovereign, provenance-preserving.

Universal Context runs a daemon that watches your AI coding sessions (Claude Code, Codex CLI, Gemini CLI), captures every turn with full provenance, summarizes them with LLM + local embeddings, and stores everything in a SurrealDB graph database. You get semantic search, keyword search, working memory distillation, and LLM-powered Q&A across your entire AI coding history.

## Install

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mysticaltech/universal-context.git
cd universal-context
uv tool install --editable . --with anthropic
uc init
```

This puts `uc` on your PATH and creates `~/.uc/config.yaml`.

For development:

```bash
uv venv && uv pip install -e ".[dev,llm]"
pytest
```

## Usage

```bash
# Start the daemon (watches sessions, summarizes turns, generates embeddings)
uc daemon start -f

# Search your history
uc search "auth middleware"              # keyword search (BM25)
uc search "auth" --project . --semantic  # semantic search, scoped to project

# Get project context (for agents)
uc context --json --context "debugging the auth middleware"

# Ask questions about your project
uc ask "How does the Gemini adapter discover sessions?" --project .
uc ask "What bugs were fixed recently?" --project .

# Working memory
uc memory show --project .      # view distilled project context
uc memory refresh --project .   # regenerate from latest sessions

# Inspect
uc status                       # system overview
uc timeline                     # turns in latest run
uc inspect turn:abc123          # full provenance chain
uc doctor                       # health check
uc dashboard                    # interactive TUI
```

Every command supports `--json` for machine-readable output.

## How It Works

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
              │ Graph DB       │
              │ file:// or ws: │
              └────────────────┘
```

**Adapters** discover sessions from each AI CLI's local files. **Triggers** detect turn boundaries in each format (JSONL, event-stream, JSON arrays). The **watcher** ingests turns into the graph DB. The **worker** processes jobs: LLM summarization with extractive fallback, embedding via EmbeddingGemma-300M (local, no API keys), and working memory distillation.

Everything is linked via graph edges (`RELATE`). Summaries point to transcripts, transcripts to turns, turns to runs, runs to scopes. Full provenance, always traversable.

## Configuration

`~/.uc/config.yaml`:

```yaml
# LLM for summarization (extractive fallback if no key)
llm_provider: "openrouter"     # openrouter | claude | openai | auto
llm_model: ""                  # empty = provider default

# Embeddings (local by default — no API key needed)
embed_provider: "local"        # local | openai | openrouter | auto

# API keys (or use env vars: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
openrouter_api_key: ""

# SurrealDB server (optional — enables BM25 + HNSW KNN)
db_url: "ws://127.0.0.1:8000"
db_user: "root"
db_pass: "root"

# Which CLIs to watch
auto_detect_claude: true
auto_detect_codex: true
auto_detect_gemini: false
```

The local embedding model (EmbeddingGemma-300M, 768 dims, ONNX Runtime) downloads ~600MB on first use and caches at `~/.cache/uc-models/`. No API keys, no PyTorch — just ONNX inference.

## SurrealDB

The embedded database (`file://`) handles storage and provenance. For full-text search (BM25) and vector search (HNSW KNN), run a SurrealDB v3 server:

```bash
curl -sSf https://install.surrealdb.com | sh
surreal start --bind 127.0.0.1:8000 --user root --pass root surrealkv://~/.surrealdb/data
```

Without a server, search falls back to substring matching and brute-force cosine similarity — still works, just slower at scale.

## License

MIT
