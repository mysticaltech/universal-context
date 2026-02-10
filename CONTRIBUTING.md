# Contributing to Universal Context

**PRs over issues.** We don't use GitHub Issues for bug reports or feature requests. If you found a problem, fix it and send a pull request. If you want to add a feature, send a PR with tests.

## Setup

```bash
git clone https://github.com/mysticaltech/universal-context.git
cd universal-context
uv venv && uv pip install -e ".[dev,llm]"
```

## PR Requirements

Every pull request must pass **triple verification** before merge:

### 1. Tests Pass

```bash
pytest -x -v
```

All 283+ tests must pass. If you're adding functionality, add tests for it.

### 2. Lint Clean

```bash
ruff check . && ruff format --check .
```

Zero warnings, zero formatting issues.

### 3. Manual Verification

Test your change end-to-end. If you changed search, run a search. If you changed the daemon, start the daemon and watch it process a session. Screenshots or terminal output in the PR description are appreciated.

## Code Conventions

- **Python 3.13+**, type hints on all function signatures
- `from __future__ import annotations` in every file
- **Pydantic v2** for data models, `StrEnum` for enumerations
- **Async by default** for DB and daemon code
- **No silent failures** — always surface errors clearly
- Secret redaction applied to all captured content
- `ruff` for linting and formatting (config in `pyproject.toml`)

## What Makes a Good PR

- **Small and focused** — one change per PR
- **Tested** — unit tests for logic, integration tests for end-to-end flows
- **Documented** — if you add a CLI command or config option, update the README

## What to Avoid

- Feature creep — keep PRs focused on a single concern
- Unnecessary abstractions — three similar lines of code is better than a premature abstraction
- Breaking changes without discussion — if your change affects the config format, CLI interface, or database schema, open a discussion first
- Dependencies without justification — every new dependency is a maintenance burden

## Architecture

The full architecture guide lives in [CLAUDE.md](CLAUDE.md). Key pointers:

- **Adapter/Trigger pattern** — each AI CLI has an adapter (session discovery) and trigger (turn detection)
- **Graph provenance** — `RELATE` edges link scope → run → turn → transcript → summary
- **Embedded + server modes** — all features work in both modes with automatic fallbacks
- **Git-aware scopes** — canonical_id from git remote URL; worktrees and clones share one scope

## Testing

```bash
# All tests
pytest -x -v

# Skip tests that need API keys
pytest -m "not integration"

# Single test file
pytest tests/test_search.py -v
```

Unit tests use `mem://` (embedded SurrealDB). Full search integration tests (BM25 + HNSW) require a running SurrealDB v3 server.
