---
name: pulling-context
description: >-
  Use this skill when you need to recall past work, understand prior decisions,
  resume a previous session, debug something seen before, or when the user asks
  "what did we do", "what was decided", "how does X work" about this project.
  Also use after completing significant work to persist knowledge.
---

# Universal Context — Project Memory

You have access to **Universal Context** (`uc`), an operational memory engine that captures AI coding sessions across Claude Code, Codex, and Gemini CLI. It stores every turn with full provenance and provides semantic search, keyword search, and LLM-powered Q&A.

## Fast path: ask a question

For most cases, a single command gets you the answer:

```bash
uc ask "How does the auth middleware work?" --project .
uc ask "What bugs were fixed recently?" --project .
uc ask "What's the database schema design?" --project . --json
```

This gathers working memory + scope-filtered search results, sends them to an LLM, and returns a synthesized answer with citations. Use `--json` when you need structured output.

Falls back to displaying raw search results if no LLM is configured.

## Deep reasoning: agentic exploration

For complex questions that need multi-hop reasoning across sessions:

```bash
uc reason "How did the auth system evolve?" --project .
uc reason "What led to choosing SurrealDB?" --project . --verbose
uc reason "Trace the scope implementation" --project . --json
```

This uses DSPy's RLM (Recursive Language Model) — the LLM writes Python in a REPL loop to explore the graph database, calling tools like `get_working_memory()`, `search_sessions()`, `search_semantic()`, `list_recent_runs()`, `get_run_turns()`, and `query_graph()`. Much more thorough than `uc ask` for questions requiring multi-step exploration.

Requires: `pip install universal-context[reason]` (installs DSPy).

## Working memory

Working memory is an LLM-distilled summary of the project's recent sessions. It may already be in your system context as a `# Project Memory:` block — use it as your baseline.

```bash
uc memory show --project .       # view current working memory
uc memory refresh --project .    # regenerate from latest sessions
```

If the memory looks stale (mentions things as "in progress" that are clearly done), refresh it.

## Deep exploration

When you need to dig deeper than a single question:

### Semantic search — find relevant past work by meaning

```bash
uc context --json --context "describe what you're working on"
```

The `--context` flag uses embedding-based retrieval. Be specific — the more detail, the better the results:
- `--context "debugging JWT token expiration in the auth middleware"`
- `--context "adding pagination to the /api/users endpoint"`

Returns `semantic_results` (ranked by cosine similarity) and `runs` (recent sessions with turn summaries).

### Branch-filtered context

```bash
uc context --project . --branch main --json         # only runs from main branch
uc context --project . --branch feature/auth --json # only runs from a feature branch
```

Runs now carry `branch`, `commit_sha`, and `merged_to` (set when a feature branch's commits are ancestors of the current branch). Branch filtering is useful when you want context from a specific line of work.

### Keyword search — find exact terms

```bash
uc context --json --query "authentication"
uc context --json --context "auth flow" --query "JWT"  # combine both
```

### Scoped artifact search

```bash
uc search "database migration" --project .             # keyword search, scoped to project
uc search "schema" --project . --kind summary --json   # filter by artifact kind
```

### Inspect a specific turn

```bash
uc inspect --json "turn:abc123"
```

Returns the turn's artifacts (transcripts, summaries) and full provenance chain.

## After significant work

Persist what was accomplished so future sessions benefit:

```bash
uc memory refresh --project .
```

This regenerates the working memory from recent sessions, backfills scope metadata (including git-aware canonical IDs), and rebuilds the vector index. Do this after completing a feature, fixing a major bug, or making architectural decisions.

## Git-aware scope resolution

UC uses git remote URLs as canonical scope identity. This means:
- **Worktrees** of the same repo share one scope (same knowledge base)
- **Clones** at different paths share one scope (if same remote)
- **`--project .`** works from any worktree or clone path
- **Non-git directories** use `path://` fallback (still works normally)

To inspect or manage scopes:

```bash
uc scope list --json          # shows canonical_id alongside name/path
uc scope show scope:abc123    # shows canonical_id, path, name
uc scope update-path <ref> /new/path  # auto-recomputes canonical_id, auto-merges if duplicate
uc scope cleanup --dry-run    # finds garbage scopes (skips git-backed ones)
```

To make the memory available to other AI tools (Codex, Gemini):

```bash
uc memory inject --project .     # writes to AGENTS.md
```

## Error handling

| Error | Action |
|-------|--------|
| `"error": "no_scope"` | No sessions tracked for this project yet |
| `"error": "no_embed_provider"` | Embeddings unconfigured — use `--query` keyword search instead |
| `"error": "embed_failed"` | Embedding API error — fall back to `--query` |
| `"error": "db_unavailable"` | Run `uc doctor` to diagnose |
| `command not found: uc` | UC CLI not installed |
| Empty results | No sessions captured — daemon may need `uc daemon start -f` |

## CLI reference

```
# Questions
uc ask "question" --project .                    # LLM-powered Q&A (best for specific questions)
uc ask "question" --project . --json             # JSON output
uc reason "question" --project .                 # Agentic multi-hop reasoning (DSPy RLM)
uc reason "question" --project . --verbose       # Show REPL trajectory
uc reason "question" --project . --json          # JSON output with trajectory

# Working memory
uc memory show --project .                       # View distilled project context
uc memory refresh --project .                    # Regenerate + backfill + rebuild index
uc memory inject --project .                     # Write to AGENTS.md (cross-IDE)
uc memory install-hook                           # Auto-inject on session start

# Search & context
uc context --json --context "task description"   # Semantic retrieval
uc context --json --query "keyword"              # Keyword search (BM25)
uc context --json --context "..." --query "..."  # Hybrid (both combined)
uc context --project . --branch main --json      # Branch-filtered context
uc search "query" --project .                    # Scoped artifact search
uc search "query" --project . --kind summary     # Filter by kind

# Inspect
uc inspect --json "turn:id"                      # Turn details + provenance
uc timeline --json                               # Latest run timeline
uc timeline --branch main --json                 # Timeline filtered by branch
uc status --json                                 # System overview

# Sharing
uc share export run:abc -o bundle.json           # Export run as v2 bundle
uc share import bundle.json --project .          # Import, matching scope by canonical_id
```
