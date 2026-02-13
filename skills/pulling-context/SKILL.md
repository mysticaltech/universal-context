---
name: pulling-context
description: >-
  Use this skill when you need to recall past work, understand prior decisions,
  resume a previous session, debug something seen before, or when the user asks
  "what did we do", "what was decided", "how does X work" about this project.
  Also use after completing significant work to persist knowledge.
---

# Universal Context — Project Memory

Use **Universal Context** (`uc`) to recover and synthesize prior project work.

## Fast path (default)

For most questions, use `ask` directly (or intent-first invocation):

```bash
uc "How does auth middleware work?"
uc ask "What changed in scope identity?" --project .
uc ask "What did we decide about SurrealDB indexes?" --project . --json
```

## Deep path (multi-hop)

Use deep reasoning only when shallow retrieval is insufficient:

```bash
uc ask "How did this subsystem evolve across sessions?" --project . --deep
uc ask "Trace the root cause timeline" --project . --deep --verbose
uc ask "Map decisions and open risks" --project . --deep --json
```

Operational notes:
- Deep mode is slower (often 30–90s+).
- Use long timeouts when running via shell tools (`120000ms`+).
- Deep mode uses UC-configured LLM provider/model (`llm_provider`, `llm_model`).
- Legacy equivalent exists at `uc admin reason`, but prefer `uc ask --deep`.

## Retrieval-first workflow

When you need to gather raw evidence before answering:

```bash
uc find "JWT expiration bug" --project . --mode auto --json
uc find "scope canonical_id" --project . --mode keyword --json
uc find "merge detection" --project . --mode semantic --json
uc find "index rebuild" --project . --mode hybrid --json
```

For branch-filtered context or legacy operator views:

```bash
uc admin context --project . --branch main --json
uc admin timeline --branch main --json
uc admin inspect turn:abc123 --json
```

## Working memory ops

```bash
uc memory show --project .
uc memory refresh --project .
uc memory sync --project .
uc memory migrate-db --project .
uc memory remember "note" --project . --type durable_fact
uc memory history --project .
```

Use `refresh` when memory looks stale. Use `sync` when you also want injection into `AGENTS.md`/`CLAUDE.md`.

Durability-first recovery commands:

```bash
uc admin db rebuild --json
uc admin db prove --project . --json
```

## After significant work

Always persist state:

```bash
uc memory refresh --project .
```

## Scope + sharing utilities

```bash
uc admin scope list --json
uc admin scope show <ref>
uc admin scope update-path <ref> /new/path
uc admin scope cleanup --dry-run

uc admin share export run:abc -o bundle.json
uc admin share import bundle.json --project .
```

## Error handling

| Error | Action |
|-------|--------|
| `no_scope` | No tracked sessions yet for this project |
| `no_embed_provider` / `embed_failed` | Use `uc find --mode keyword` |
| DB unavailable | Run `uc doctor` |
| Empty results | Ensure daemon is running and sessions were captured |

## Current CLI reference

```bash
# Public surface
uc --help
uc doctor
uc status
uc find "query" --project .
uc ask "question" --project .
uc ask "question" --project . --deep
uc remember "fact" --project . --type durable_fact
uc daemon start -f
uc memory --help

# Admin surface
uc admin --help
uc admin db --help
uc admin context --project . --branch main --json
uc admin reason "question" --verbose
uc admin timeline --json
uc admin inspect turn:abc123 --json
uc admin rebuild-index
```
