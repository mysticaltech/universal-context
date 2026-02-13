"""Universal Context CLI — main entry point."""

from __future__ import annotations

import asyncio
import json as json_mod
import os
import signal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, get_uc_home


class UCApp(typer.Typer):
    """Typer app with intent-first routing: `uc \"question\"` -> `uc ask \"question\"`."""

    _DIRECT_COMMANDS = {
        "doctor",
        "status",
        "find",
        "ask",
        "remember",
        "daemon",
        "memory",
        "admin",
    }

    def __call__(self, *args, **kwargs):
        import sys

        argv = sys.argv[1:]
        if argv and argv[0] not in self._DIRECT_COMMANDS and not argv[0].startswith("-"):
            original_argv = sys.argv[:]
            sys.argv = [sys.argv[0], "ask", *argv]
            try:
                return super().__call__(*args, **kwargs)
            finally:
                sys.argv = original_argv
        return super().__call__(*args, **kwargs)


app = UCApp(
    name="uc",
    help="Universal Context — operational memory engine for AI agents",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

# --- Sub-command groups ---

daemon_app = typer.Typer(help="Manage the UC daemon (watcher + worker)")
app.add_typer(daemon_app, name="daemon")

memory_app = typer.Typer(help="Project working memory")
app.add_typer(memory_app, name="memory")

admin_app = typer.Typer(help="Advanced and operator-focused commands")
app.add_typer(admin_app, name="admin")

share_app = typer.Typer(help="Export and import share bundles")
admin_app.add_typer(share_app, name="share")

db_admin_app = typer.Typer(help="Rebuild and recover derived storage")
admin_app.add_typer(db_admin_app, name="db")

config_app = typer.Typer(help="View and modify configuration")
admin_app.add_typer(config_app, name="config")

scope_app = typer.Typer(help="Manage project scopes")
admin_app.add_typer(scope_app, name="scope")


# --- Async helper ---


def _run_async(coro):
    """Run an async function from sync CLI context."""
    return asyncio.run(coro)


def _get_db():
    """Get a database connection for CLI commands."""
    from .config import UCConfig
    from .db.client import UCDatabase

    config = UCConfig.load()
    if config.db_url:
        return UCDatabase.from_url(config.db_url, config.db_user, config.db_pass)
    return UCDatabase.from_path(Path(config.resolved_db_path))


# --- Serialization helper ---


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert SurrealDB RecordID objects to strings for JSON output."""
    sanitized: dict[str, Any] = {}
    for k, v in record.items():
        if hasattr(v, "__class__") and "RecordID" in type(v).__name__:
            sanitized[k] = str(v)
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_record(v)
        elif isinstance(v, list):
            sanitized[k] = [
                _sanitize_record(i)
                if isinstance(i, dict)
                else str(i)
                if hasattr(i, "__class__") and "RecordID" in type(i).__name__
                else i
                for i in v
            ]
        else:
            sanitized[k] = v
    return sanitized


def _sanitize_search_result(record: dict[str, Any]) -> dict[str, Any]:
    """Sanitize artifact/search records and drop heavy vector payloads."""
    sanitized = _sanitize_record(record)
    sanitized.pop("embedding", None)
    return sanitized


# --- Top-level commands ---


@app.callback()
def main(ctx: typer.Context) -> None:
    """UC — operational memory for AI agents."""
    if ctx.invoked_subcommand is None:
        console.print(f"[bold]Universal Context[/bold] v{__version__}")
        console.print('Ask directly: [cyan]uc "why did we choose SurrealDB?"[/cyan]')
        console.print("Run [cyan]uc --help[/cyan] for available commands.")


@admin_app.command("init")
def init() -> None:
    """Set up UC home directory (~/.uc/) and default config."""
    from .config import get_default_config_content
    from .memory_repo import bootstrap_memory_repo

    uc_home = get_uc_home()

    for subdir in ["data", "blobs", "shares", "memory"]:
        (uc_home / subdir).mkdir(parents=True, exist_ok=True)

    bootstrap_memory_repo(uc_home / "memory", init_git=True)

    config_path = uc_home / "config.yaml"
    if not config_path.exists():
        config_path.write_text(get_default_config_content(), encoding="utf-8")
        console.print(f"[green]Created config:[/green] {config_path}")
    else:
        console.print(f"[dim]Config already exists:[/dim] {config_path}")

    console.print(f"[green]UC home ready:[/green] {uc_home}")


@admin_app.command("version")
def version() -> None:
    """Show version information."""
    console.print(f"Universal Context v{__version__}")


@app.command()
def doctor() -> None:
    """Health check — verify DB, adapters, and daemon status."""
    import shutil

    console.print("[bold]Universal Context Doctor[/bold]\n")

    uc_home = get_uc_home()
    _check("UC home exists", uc_home.exists())

    config_path = uc_home / "config.yaml"
    _check("Config file exists", config_path.exists())

    try:
        import surrealdb  # noqa: F401

        _check("SurrealDB SDK installed", True)
    except ImportError:
        _check("SurrealDB SDK installed", False)

    _check("tmux available", shutil.which("tmux") is not None)

    from .config import UCConfig

    config = UCConfig.load()
    keys = {
        "OpenRouter": config.get_api_key("openrouter"),
        "OpenAI": config.get_api_key("openai"),
        "Anthropic": config.get_api_key("anthropic"),
    }
    has_any = any(keys.values())
    for name, key in keys.items():
        if key:
            _check(f"{name} API key", True)
        elif has_any:
            console.print(f"  [dim]SKIP[/dim]  {name} API key")
        else:
            _check(f"{name} API key", False)

    pid_file = uc_home / "daemon.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        console.print(f"  [cyan]Daemon PID:[/cyan] {pid}")
    else:
        console.print("  [dim]Daemon: not running[/dim]")



@admin_app.command("timeline")
def timeline(
    run_id: str | None = typer.Argument(None, help="Run ID (omit for latest)"),
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Project path to scope the query"
    ),
    branch: str | None = typer.Option(
        None, "--branch", "-b", help="Filter runs by git branch (used when no run_id)"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show chronological timeline of a run."""

    async def _timeline():
        from .db.queries import list_runs, list_turns
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            scope_id = None
            if project is not None and run_id is None:
                scope = await _resolve_scope(db, str(project.resolve()))
                if scope:
                    scope_id = str(scope["id"])

            if run_id is None:
                runs = await list_runs(
                    db,
                    scope_id=scope_id,
                    branch=branch,
                    limit=1,
                )
                if not runs:
                    if json_output:
                        print(json_mod.dumps({"error": "no_runs", "message": "No runs found"}))
                    else:
                        console.print("[dim]No runs found.[/dim]")
                    return
                run_record = runs[0]
                rid = str(run_record["id"])
            else:
                rid = run_id
                from .db.queries import get_run

                run_record = await get_run(db, rid)

            turns = await list_turns(db, rid)

            if json_output:
                out: dict[str, Any] = {"run_id": rid}
                if run_record:
                    out["branch"] = run_record.get("branch")
                    out["commit_sha"] = run_record.get("commit_sha")
                    out["merged_to"] = run_record.get("merged_to")
                    out["status"] = run_record.get("status")
                out["turns"] = [_sanitize_record(t) for t in turns]
                print(json_mod.dumps(out, default=str))
                return

            if not turns:
                console.print(f"[dim]No turns in run {rid}.[/dim]")
                return

            table = Table(title=f"Timeline: {rid}")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("User Message", max_width=50)
            table.add_column("Started", style="dim")
            for t in turns:
                msg = (t.get("user_message") or "")[:50]
                ts = str(t.get("started_at", ""))[:19]
                table.add_row(str(t.get("sequence", "")), msg, ts)
            console.print(table)
        finally:
            await db.close()

    _run_async(_timeline())


@admin_app.command("inspect")
def inspect(
    turn_id: str = typer.Argument(..., help="Turn ID to inspect"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show turn details, artifacts, and provenance."""

    async def _inspect():
        from .db.queries import get_provenance_chain, get_turn, get_turn_artifacts
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            turn = await get_turn(db, turn_id)
            if not turn:
                if json_output:
                    print(json_mod.dumps({"error": "not_found", "turn_id": turn_id}))
                    return
                console.print(f"[red]Turn not found:[/red] {turn_id}")
                raise typer.Exit(code=1)

            artifacts = await get_turn_artifacts(db, turn_id)
            chain = await get_provenance_chain(db, turn_id)

            if json_output:
                # Flatten artifact IDs from graph traversal
                artifact_ids = []
                for a in artifacts or []:
                    produced = a.get("->produced", {})
                    if isinstance(produced, dict):
                        for aid in produced.get("->artifact", []):
                            artifact_ids.append(str(aid))

                print(
                    json_mod.dumps(
                        {
                            **_sanitize_record(turn),
                            "artifacts": artifact_ids,
                            "provenance": [_sanitize_record(c) for c in chain] if chain else [],
                        },
                        default=str,
                    )
                )
                return

            console.print(f"[bold]Turn {turn_id}[/bold]\n")
            console.print(f"  Run: {turn.get('run')}")
            console.print(f"  Sequence: {turn.get('sequence')}")
            console.print(f"  User: {turn.get('user_message', '(none)')}")
            console.print(f"  Started: {turn.get('started_at')}")

            if artifacts:
                console.print("\n[bold]Artifacts:[/bold]")
                for a in artifacts:
                    produced = a.get("->produced", {})
                    ids = produced.get("->artifact", [])
                    for aid in ids:
                        console.print(f"  - {aid}")

            if chain:
                console.print("\n[bold]Provenance:[/bold]")
                console.print(f"  {chain}")
        finally:
            await db.close()

    _run_async(_inspect())


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show overview: scopes, runs, job counts."""

    async def _status():
        from .db.queries import count_jobs_by_status, list_runs, list_scopes
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            scopes = await list_scopes(db)
            runs = await list_runs(db, limit=5)
            jobs = await count_jobs_by_status(db)

            if json_output:
                print(
                    json_mod.dumps(
                        {
                            "scopes": [_sanitize_record(s) for s in scopes],
                            "recent_runs": [_sanitize_record(r) for r in runs],
                            "jobs": jobs,
                        },
                        default=str,
                    )
                )
                return

            console.print(f"[bold]Scopes:[/bold] {len(scopes)}")
            for s in scopes[:5]:
                console.print(f"  {s['id']}  {s['name']}  {s.get('path', '')}")

            console.print(f"\n[bold]Recent runs:[/bold] {len(runs)}")
            for r in runs:
                console.print(
                    f"  {r['id']}  {r.get('agent_type', '')}  "
                    f"[{'green' if r.get('status') == 'active' else 'dim'}]"
                    f"{r.get('status', '')}[/]"
                )

            if jobs:
                console.print(f"\n[bold]Jobs:[/bold] {jobs}")
        finally:
            await db.close()

    _run_async(_status())


@admin_app.command("context")
def context(
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Project path (default: cwd)"
    ),
    query: str | None = typer.Option(None, "--query", "-q", help="Keyword search query"),
    ctx: str | None = typer.Option(
        None,
        "--context",
        "-c",
        help="Semantic context — describe what you're working on for embedding-based retrieval",
    ),
    branch: str | None = typer.Option(None, "--branch", "-b", help="Filter runs by git branch"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max runs to show"),
    turns: int = typer.Option(10, "--turns", "-t", help="Max turns per run"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Pull context from past sessions for the current project.

    Use --context for semantic search (embedding-based), --query for keyword search (BM25).
    Both can be combined. --context requires an OpenAI or OpenRouter API key.
    """

    async def _context():
        from .db.queries import (
            get_turn_summaries,
            list_runs,
            search_artifacts,
        )
        from .db.schema import apply_schema

        project_path = str((project or Path(".")).resolve())

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            scope = await _resolve_scope(db, project_path)
            if not scope:
                payload = {
                    "error": "no_scope",
                    "project": project_path,
                    "message": f"No scope found for {project_path}",
                }
                if json_output:
                    print(json_mod.dumps(payload))
                else:
                    console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                return

            scope_id = str(scope["id"])
            runs_data = await list_runs(
                db,
                scope_id=scope_id,
                branch=branch,
                limit=limit,
            )

            # Build run details with summaries
            run_details = []
            for r in runs_data:
                rid = str(r["id"])
                summaries = await get_turn_summaries(db, rid, limit=turns)
                run_details.append(
                    {
                        "run_id": rid,
                        "agent_type": r.get("agent_type", ""),
                        "status": r.get("status", ""),
                        "started_at": r.get("started_at"),
                        "branch": r.get("branch"),
                        "commit_sha": r.get("commit_sha"),
                        "merged_to": r.get("merged_to"),
                        "total_turns": len(summaries),
                        "turns": summaries,
                    }
                )

            # Optional keyword search (scope-filtered)
            keyword_results = []
            if query:
                raw = await search_artifacts(
                    db,
                    query,
                    kind="summary",
                    scope_id=scope_id,
                )
                keyword_results = [_sanitize_record(sr) for sr in raw]

            # Optional semantic search (scope-filtered)
            semantic_results = []
            if ctx:
                semantic_results = await _semantic_search(
                    db,
                    ctx,
                    query,
                    scope_id=scope_id,
                )

            if json_output:
                print(
                    json_mod.dumps(
                        {
                            "scope": _sanitize_record(scope),
                            "runs": run_details,
                            "search_results": keyword_results,
                            "semantic_results": semantic_results,
                        },
                        default=str,
                    )
                )
                return

            # Rich output
            console.print(
                f"[bold]Scope:[/bold] {scope.get('name')} [dim]({scope.get('path')})[/dim]\n"
            )

            if not run_details:
                console.print("[dim]No runs found for this scope.[/dim]")
                return

            for rd in run_details:
                console.print(
                    f"[bold]{rd['run_id']}[/bold]  "
                    f"{rd['agent_type']}  "
                    f"[{'green' if rd['status'] == 'active' else 'dim'}]"
                    f"{rd['status']}[/]"
                )
                for t in rd["turns"]:
                    msg = (t.get("user_message") or "")[:40]
                    summary = (t.get("summary") or "[no summary]")[:60]
                    console.print(f"  [cyan]#{t['sequence']}[/cyan] {msg}")
                    console.print(f"    [dim]{summary}[/dim]")
                console.print()

            if keyword_results:
                console.print(f"[bold]Keyword results for '{query}':[/bold]")
                for sr in keyword_results:
                    console.print(f"  {sr.get('id')}  {sr.get('content', '')[:60]}")

            if semantic_results:
                console.print(
                    f"\n[bold]Semantic matches for context:[/bold] [dim]({ctx[:40]}...)[/dim]"
                    if ctx and len(ctx) > 40
                    else "\n[bold]Semantic matches:[/bold]"
                )
                for sr in semantic_results:
                    score = sr.get("score", 0)
                    content = sr.get("content", "")[:60]
                    console.print(f"  [cyan]{score:.3f}[/cyan]  {sr.get('id')}  {content}")

        finally:
            await db.close()

    _run_async(_context())


async def _semantic_search(
    db: Any,
    context_text: str,
    keyword_query: str | None = None,
    scope_id: str | None = None,
    kind: str | None = "summary",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Embed context text and run hybrid or vector-only search.

    If a keyword_query is also provided, uses hybrid RRF to fuse
    text + vector results. Otherwise uses vector-only search.
    """
    from .config import UCConfig
    from .db.queries import hybrid_search, semantic_search

    config = UCConfig.load()

    from .embed import create_embed_provider

    provider = await create_embed_provider(config)
    if provider is None:
        return [{"error": "no_embed_provider", "message": "Embeddings disabled in config"}]

    try:
        query_embedding = await provider.embed_query(context_text)

        if keyword_query:
            results = await hybrid_search(
                db,
                keyword_query,
                query_embedding,
                kind=kind,
                limit=limit,
                scope_id=scope_id,
            )
        else:
            results = await semantic_search(
                db,
                query_embedding,
                kind=kind,
                limit=limit,
                scope_id=scope_id,
            )

        return [_sanitize_search_result(r) for r in results]
    except Exception as e:
        return [{"error": "embed_failed", "message": str(e)}]


@app.command()
def find(
    text: str = typer.Argument(..., help="Search text or semantic context"),
    mode: str = typer.Option(
        "auto",
        "--mode",
        "-m",
        help="Search mode: auto, keyword, semantic, hybrid",
    ),
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Scope search to a project path"
    ),
    kind: str | None = typer.Option(None, "--kind", "-k", help="Filter by artifact kind"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Unified retrieval command for keyword and semantic memory search."""

    async def _find():
        from .db.queries import search_artifacts
        from .db.schema import apply_schema

        normalized_mode = mode.strip().lower()
        valid_modes = {"auto", "keyword", "semantic", "hybrid"}
        if normalized_mode not in valid_modes:
            console.print(
                f"[red]Invalid mode:[/red] {mode}. "
                f"Use one of: {', '.join(sorted(valid_modes))}"
            )
            raise typer.Exit(code=1)

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            scope_id = None
            if project is not None:
                project_path = str(project.resolve())
                scope = await _resolve_scope(db, project_path)
                if scope:
                    scope_id = str(scope["id"])

            keyword_results: list[dict[str, Any]] = []
            semantic_results: list[dict[str, Any]] = []

            if normalized_mode in {"auto", "keyword", "hybrid"}:
                raw_keyword = await search_artifacts(
                    db,
                    text,
                    kind=kind,
                    limit=limit,
                    scope_id=scope_id,
                )
                keyword_results = [_sanitize_search_result(r) for r in raw_keyword]

            if normalized_mode in {"auto", "semantic", "hybrid"}:
                semantic_query = text if normalized_mode == "hybrid" else None
                semantic_results = await _semantic_search(
                    db,
                    text,
                    keyword_query=semantic_query,
                    scope_id=scope_id,
                    kind=kind,
                    limit=limit,
                )

            valid_semantic = [r for r in semantic_results if "error" not in r]
            semantic_errors = [r for r in semantic_results if "error" in r]

            if normalized_mode == "keyword":
                primary_results = keyword_results
            elif normalized_mode in {"semantic", "hybrid"}:
                primary_results = valid_semantic
            else:
                primary_results = valid_semantic if valid_semantic else keyword_results

            if json_output:
                print(
                    json_mod.dumps(
                        {
                            "query": text,
                            "mode": normalized_mode,
                            "scope": scope_id,
                            "results": primary_results,
                            "keyword_results": keyword_results,
                            "semantic_results": valid_semantic,
                            "semantic_errors": semantic_errors,
                        },
                        default=str,
                    )
                )
                return

            if not primary_results:
                console.print("[dim]No results found.[/dim]")
                if semantic_errors:
                    console.print(f"[dim]Semantic issue: {semantic_errors[0].get('message')}[/dim]")
                return

            table = Table(title=f"Find ({normalized_mode}): {text}")
            table.add_column("ID", style="cyan")
            table.add_column("Kind", style="green")
            table.add_column("Score", style="yellow", justify="right")
            table.add_column("Content", max_width=70)
            for r in primary_results[:limit]:
                content = (r.get("content") or "")[:70]
                score = r.get("score")
                score_text = f"{score:.3f}" if isinstance(score, float) else "-"
                table.add_row(str(r.get("id", "")), r.get("kind", ""), score_text, content)
            console.print(table)
        finally:
            await db.close()

    _run_async(_find())


async def _run_reasoning(
    question: str,
    project_path: str,
    max_iterations: int,
    max_llm_calls: int,
    verbose: bool,
) -> dict[str, Any]:
    from .config import UCConfig
    from .reason import reason as run_reason

    config = UCConfig.load()
    return await run_reason(
        question=question,
        project_path=project_path,
        config=config,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        verbose=verbose,
    )


async def _persist_reasoning_snapshot(
    project_path: str,
    result: dict[str, Any],
) -> str | None:
    """Persist deep-reasoning structured fields onto working-memory metadata."""
    from .db.queries import set_working_memory_reasoning_metadata
    from .db.schema import apply_schema

    db = _get_db()
    await db.connect()
    try:
        await apply_schema(db)
        scope = await _resolve_scope(db, project_path)
        if not scope:
            return None

        scope_id = str(scope["id"])
        reasoning = {
            "answer_preview": (result.get("answer") or "")[:500],
            "facts": result.get("facts", []),
            "decisions": result.get("decisions", []),
            "open_questions": result.get("open_questions", []),
            "evidence_ids": result.get("evidence_ids", []),
        }
        return await set_working_memory_reasoning_metadata(db, scope_id, reasoning)
    finally:
        await db.close()


def _render_reason_output(result: dict[str, Any], verbose: bool) -> None:
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print(
        Panel(
            Markdown(result["answer"]),
            title="Answer",
            border_style="green",
        )
    )

    llm_provider = result.get("llm_provider")
    llm_model = result.get("llm_model")
    if llm_provider and llm_model:
        console.print(f"[dim]LLM: {llm_provider} ({llm_model})[/dim]")

    facts = result.get("facts") or []
    decisions = result.get("decisions") or []
    open_questions = result.get("open_questions") or []
    evidence_ids = result.get("evidence_ids") or []

    if facts:
        console.print("\n[bold]Facts[/bold]")
        for item in facts:
            console.print(f"- {item}")

    if decisions:
        console.print("\n[bold]Decisions[/bold]")
        for item in decisions:
            console.print(f"- {item}")

    if open_questions:
        console.print("\n[bold]Open Questions[/bold]")
        for item in open_questions:
            console.print(f"- {item}")

    if evidence_ids:
        console.print("\n[bold]Evidence IDs[/bold]")
        for item in evidence_ids:
            console.print(f"- [cyan]{item}[/cyan]")

    if verbose and result.get("trajectory"):
        console.print("\n[bold]Trajectory:[/bold]")
        for i, step in enumerate(result["trajectory"], 1):
            if isinstance(step, dict):
                code = step.get("code", "")
                output = step.get("output", "")
                console.print(f"\n[cyan]Step {i}:[/cyan]")
                if code:
                    from rich.syntax import Syntax

                    console.print(Syntax(str(code), "python", theme="monokai"))
                if output:
                    console.print(f"[dim]{str(output)[:500]}[/dim]")
            else:
                console.print(f"\n[cyan]Step {i}:[/cyan] {str(step)[:500]}")

    scope_info = f"scope: {result.get('scope', 'none')}"
    iterations = result.get("iterations")
    iter_info = f"  iterations: {iterations}" if iterations else ""
    console.print(f"\n[dim]{scope_info}{iter_info}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the project"),
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Project path (default: cwd)"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Max search results for context"),
    deep: bool = typer.Option(False, "--deep", help="Use agentic deep reasoning"),
    auto_deep: bool = typer.Option(
        True,
        "--auto-deep/--no-auto-deep",
        help="Auto-escalate to deep reasoning when shallow context is missing",
    ),
    max_iterations: int = typer.Option(12, "--max-iterations", help="Max deep REPL iterations"),
    max_llm_calls: int = typer.Option(30, "--max-llm-calls", help="Max deep LLM calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show deep reasoning trajectory"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Ask a question about a project. Use --deep for agentic reasoning."""

    async def _ask():
        from .config import UCConfig
        from .db.queries import get_working_memory, search_artifacts
        from .db.schema import apply_schema
        from .git import resolve_canonical_id
        from .llm import ASK_SYSTEM_PROMPT, create_llm_fn
        from .memory_repo import (
            MEMORY_SECTIONS,
            list_scope_sections,
            render_section_text,
        )

        project_path = str((project or Path(".")).resolve())
        project_dir = Path(project_path)

        if deep:
            if not json_output:
                console.print("[dim]Reasoning... (this may take a moment)[/dim]")
            result = await _run_reasoning(
                question=question,
                project_path=project_path,
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                verbose=verbose,
            )
            persisted_memory_id = await _persist_reasoning_snapshot(project_path, result)
            if json_output:
                payload = dict(result)
                payload["mode"] = "deep"
                payload["persisted_working_memory"] = persisted_memory_id
                print(json_mod.dumps(payload, default=str))
            else:
                _render_reason_output(result, verbose=verbose)
                if persisted_memory_id:
                    console.print(
                        f"[dim]Updated reasoning metadata on {persisted_memory_id}[/dim]"
                    )
            return

        config = UCConfig.load()

        context_parts: list[str] = []
        search_results: list[dict[str, Any]] = []
        valid_semantic: list[dict[str, Any]] = []
        scope_id: str | None = None

        canonical_id = ""
        try:
            canonical_id = resolve_canonical_id(project_dir)
        except Exception:
            canonical_id = ""

        if canonical_id:
            repo_sections = list_scope_sections(
                canonical_id=canonical_id,
                display_name=project_dir.name,
                scope_path=project_path,
            )
            repo_parts = []
            for section in MEMORY_SECTIONS:
                entries = repo_sections.get(section, [])
                rendered = render_section_text(entries)
                if rendered:
                    title = section.replace("_", " ").title()
                    repo_parts.append(f"## {title}\n{rendered}")
            if repo_parts:
                context_parts.append("\n\n".join(repo_parts))

        db = None
        db_connected = False
        try:
            db = _get_db()
            await db.connect()
            db_connected = True
            await apply_schema(db)

            scope = await _resolve_scope(db, project_path)
            scope_id = str(scope["id"]) if scope else None

            if scope_id and not context_parts:
                scope_canonical = scope.get("canonical_id") if scope else None
                if scope and not scope_canonical and scope.get("path"):
                    scope_canonical = resolve_canonical_id(Path(scope["path"]))

                scope_repo_parts = []
                if scope_canonical:
                    sections = list_scope_sections(
                        canonical_id=scope_canonical,
                        display_name=str(scope.get("name", project_dir.name)),
                        scope_path=str(scope.get("path", project_path)),
                    )
                    for section in MEMORY_SECTIONS:
                        entries = sections.get(section, [])
                        rendered = render_section_text(entries)
                        if rendered:
                            title = section.replace("_", " ").title()
                            scope_repo_parts.append(f"## {title}\n{rendered}")

                if scope_repo_parts:
                    context_parts.append("\n\n".join(scope_repo_parts))
                else:
                    memory = await get_working_memory(db, scope_id)
                    if memory and memory.get("content"):
                        context_parts.append(f"## Working Memory\n{memory['content']}")

            search_results = await search_artifacts(
                db,
                question,
                kind="summary",
                limit=limit,
                scope_id=scope_id,
            )
            if search_results:
                summaries_text = "\n".join(
                    f"- {r.get('content', '')[:500]}" for r in search_results
                )
                context_parts.append(f"## Relevant Session Summaries\n{summaries_text}")

            semantic_results = await _semantic_search(
                db,
                question,
                scope_id=scope_id,
            )
            valid_semantic = [r for r in semantic_results if "error" not in r]
            if valid_semantic:
                sem_text = "\n".join(
                    f"- {r.get('content', '')[:500]}" for r in valid_semantic
                )
                context_parts.append(f"## Semantic Matches\n{sem_text}")
        except Exception as exc:
            if not context_parts:
                if json_output:
                    print(
                        json_mod.dumps(
                            {
                                "error": "db_unavailable",
                                "message": str(exc),
                            }
                        )
                    )
                else:
                    console.print("[yellow]DB unavailable and no durable memory found.[/yellow]")
                return
            if not json_output:
                console.print("[dim]DB unavailable; answering from durable memory only.[/dim]")
        finally:
            if db_connected and db is not None:
                await db.close()

        if not context_parts:
            if auto_deep:
                if not json_output:
                    console.print(
                        "[dim]Shallow context missing. Escalating to deep reasoning...[/dim]"
                    )
                result = await _run_reasoning(
                    question=question,
                    project_path=project_path,
                    max_iterations=max_iterations,
                    max_llm_calls=max_llm_calls,
                    verbose=verbose,
                )
                persisted_memory_id = await _persist_reasoning_snapshot(project_path, result)
                if json_output:
                    payload = dict(result)
                    payload["mode"] = "deep_auto"
                    payload["persisted_working_memory"] = persisted_memory_id
                    print(json_mod.dumps(payload, default=str))
                else:
                    _render_reason_output(result, verbose=verbose)
                    if persisted_memory_id:
                        console.print(
                            f"[dim]Updated reasoning metadata on {persisted_memory_id}[/dim]"
                        )
                return

            payload = {
                "error": "no_context",
                "message": "No project context found. Run the daemon first.",
            }
            if json_output:
                print(json_mod.dumps(payload))
            else:
                console.print("[yellow]No project context found.[/yellow]")
                console.print("[dim]Run the daemon to capture sessions first.[/dim]")
            return

        context_block = "\n\n".join(context_parts)
        prompt = f"## Project Context\n{context_block}\n\n## Question\n{question}"

        llm_fn = await create_llm_fn(
            config,
            system_prompt=ASK_SYSTEM_PROMPT,
            max_tokens=1000,
        )

        if llm_fn is None:
            if json_output:
                print(
                    json_mod.dumps(
                        {
                            "answer": None,
                            "context": context_block,
                            "message": "No LLM configured. Showing raw context.",
                        },
                        default=str,
                    )
                )
            else:
                console.print("[yellow]No LLM configured — showing raw context:[/yellow]\n")
                from rich.markdown import Markdown

                console.print(Markdown(context_block))
            return

        if not json_output:
            console.print("[dim]Thinking...[/dim]")

        answer = await llm_fn(prompt)

        if json_output:
            print(
                json_mod.dumps(
                    {
                        "answer": answer,
                        "mode": "shallow",
                        "sources": len(search_results) + len(valid_semantic),
                        "scope": scope_id,
                    },
                    default=str,
                )
            )
        else:
            from rich.markdown import Markdown
            from rich.panel import Panel

            console.print(
                Panel(
                    Markdown(answer),
                    title="Answer",
                    border_style="green",
                )
            )
            console.print(
                f"[dim]Sources: {len(search_results)} keyword + "
                f"{len(valid_semantic)} semantic matches[/dim]"
            )

    _run_async(_ask())


@admin_app.command("reason")
def reason(
    question: str = typer.Argument(..., help="Question to explore via agentic reasoning"),
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Project path (default: cwd)"
    ),
    max_iterations: int = typer.Option(12, "--max-iterations", help="Max REPL iterations"),
    max_llm_calls: int = typer.Option(30, "--max-llm-calls", help="Max LLM calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show step-by-step trajectory"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Legacy deep reasoning entrypoint. Prefer `uc ask --deep`."""

    async def _reason():
        project_path = str((project or Path(".")).resolve())
        if not json_output:
            console.print("[dim]Reasoning... (this may take a moment)[/dim]")
        result = await _run_reasoning(
            question=question,
            project_path=project_path,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=verbose,
        )

        if json_output:
            print(json_mod.dumps(result, default=str))
            return

        _render_reason_output(result, verbose=verbose)

    _run_async(_reason())


@admin_app.command("rebuild-index")
def rebuild_index(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Rebuild the HNSW vector index (server mode only)."""

    async def _rebuild():
        from .db.schema import apply_schema, rebuild_hnsw_index

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            rebuilt = await rebuild_hnsw_index(db)
            if json_output:
                print(json_mod.dumps({"rebuilt": rebuilt}))
            elif rebuilt:
                console.print("[green]HNSW index rebuilt successfully.[/green]")
            else:
                console.print("[dim]Skipped — HNSW index requires server mode.[/dim]")
        finally:
            await db.close()

    _run_async(_rebuild())


@db_admin_app.command("rebuild")
def rebuild_db(
    since: str | None = typer.Option(
        None,
        "--since",
        help="Replay sessions modified on/after this ISO timestamp",
    ),
    llm_missing: bool = typer.Option(
        False,
        "--llm-missing",
        help="Use LLM when a summary cache entry is missing",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore summary sidecars and always recompute summaries",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Rebuild derived DB from captured sessions and rehydrate memory projections."""

    async def _rebuild() -> None:
        from .config import UCConfig
        from .db.schema import apply_schema
        from .embed import create_embed_provider
        from .rebuild import rebuild_derived_db

        config = UCConfig.load()

        db = _get_db()
        await db.connect()
        try:
            embed_provider = await create_embed_provider(config)
            embedding_dim = embed_provider.dim if embed_provider else 768
            await apply_schema(db, embedding_dim=embedding_dim)

            try:
                result = await rebuild_derived_db(
                    db,
                    config,
                    since=since,
                    llm_missing=llm_missing,
                    no_cache=no_cache,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc))

            if json_output:
                print(json_mod.dumps(result, default=str))
                return

            console.print(f"[green]Rebuild mode:[/green] {result['mode']}")
            console.print(f"[green]Runs created:[/green] {result['runs_created']}")
            console.print(f"[green]Turns replayed:[/green] {result['turns_replayed']}")
            console.print(
                f"[green]Summary jobs processed:[/green] "
                f"{result['summary_jobs_processed']}"
            )
            if result["summary_jobs_failed"]:
                console.print(
                    f"[yellow]Summary jobs failed:[/yellow] "
                    f"{result['summary_jobs_failed']}"
                )
            console.print(
                f"[green]Working memory projections:[/green] "
                f"{result['working_memory_rehydrated']}"
            )
            console.print(f"[green]HNSW rebuilt:[/green] {result['hnsw_rebuilt']}")
        finally:
            await db.close()

    _run_async(_rebuild())


@db_admin_app.command("prove")
def prove_db(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Target one scope for proof (default: all discovered scopes).",
    ),
    skip_rebuild: bool = typer.Option(
        False,
        "--skip-rebuild",
        help="Validate file-backed continuity without rebuilding DB.",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="If rebuilding, replay sessions modified on/after this ISO timestamp.",
    ),
    llm_missing: bool = typer.Option(
        False,
        "--llm-missing",
        help="If rebuilding, use LLM on cache misses.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="If rebuilding, ignore summary sidecars and recompute summaries.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Emit machine-readable output."),
) -> None:
    """Verify durability invariants and rebuild restoration (proof command)."""

    async def _prove() -> None:
        from .config import UCConfig
        from .db.queries import (
            get_provenance_chain,
            get_working_memory,
            hybrid_search,
            list_scopes,
            search_artifacts,
        )
        from .db.schema import apply_schema
        from .embed import create_embed_provider
        from .memory_repo import normalize_canonical_id
        from .rebuild import rebuild_derived_db

        project_scopes = _collect_memory_repo_scopes(project)
        if not project_scopes:
            if project:
                raise typer.BadParameter(f"No durable memory scope found for {project}")
            raise typer.BadParameter("No durable memory scopes found in ~/.uc/memory.")

        continuity_checks: list[dict[str, Any]] = []
        probe_scopes: list[dict[str, Any]] = []
        for scope_record in project_scopes:
            sections, flat_entries, count = _gather_scope_entries(scope_record)
            probe = _derive_probe_text(flat_entries)
            scope_name = str(
                scope_record.get("scope_name")
                or scope_record.get("display_name")
                or "project"
            )
            continuity_checks.append(
                {
                    "canonical_id": scope_record["canonical_id"],
                    "scope_name": scope_name,
                    "entries": count,
                    "sections": {name: len(entries) for name, entries in sections.items()},
                    "probe": probe,
                },
            )
            if count > 0:
                probe_scopes.append(
                    {
                        "scope_record": scope_record,
                        "probe": probe,
                        "entry_count": count,
                        "scope_name": scope_name,
                    }
                )

        continuity_ok = len(continuity_checks) > 0
        rebuild_checks: list[dict[str, Any]] = []
        rebuild_ok = False
        db_rebuild_result: dict[str, Any] | None = None

        if not skip_rebuild:
            config = UCConfig.load()
            db = _get_db()
            await db.connect()
            try:
                embed_provider = await create_embed_provider(config)
                embedding_dim = embed_provider.dim if embed_provider else 768
                await apply_schema(db, embedding_dim=embedding_dim)

                try:
                    db_rebuild_result = await rebuild_derived_db(
                        db,
                        config,
                        since=since,
                        llm_missing=llm_missing,
                        no_cache=no_cache,
                    )
                except ValueError as exc:
                    raise typer.BadParameter(str(exc))

                db_scopes = await list_scopes(db)
                scope_index = _build_scope_index(db_scopes)
                db_scope_records = {str(scope["id"]): scope for scope in db_scopes}

                for item in probe_scopes:
                    scope_record = item["scope_record"]
                    probe = item["probe"]
                    canonical_id = str(scope_record["canonical_id"])
                    normalized_canonical = normalize_canonical_id(canonical_id)
                    scope_id = scope_index.get(normalized_canonical, "")
                    if not scope_id:
                        # fallback by path if canonical matching failed
                        for sid, scope in db_scope_records.items():
                            if str(scope.get("path") or "") == str(scope_record.get("path") or ""):
                                scope_id = sid
                                break

                    check = {
                        "canonical_id": canonical_id,
                        "scope_id": scope_id,
                        "scope_name": item["scope_name"],
                        "entry_count": item["entry_count"],
                        "scope_lookup": normalized_canonical,
                        "scope_found": bool(scope_id),
                        "search_scoped": False,
                        "working_memory_present": False,
                        "working_memory_method": "",
                        "provenance_ok": False,
                    }

                    if scope_id:
                        working = await get_working_memory(db, scope_id)
                        if working:
                            check["working_memory_present"] = True
                            check["working_memory_method"] = (
                                working.get("metadata", {}).get("method", "")
                            )

                        # Require a non-empty search probe only if we have repo content.
                        if item["entry_count"] > 0 and probe:
                            if embed_provider is not None and probe:
                                try:
                                    query_embedding = await embed_provider.embed_query(probe)
                                    results = await hybrid_search(
                                        db,
                                        query_text=probe,
                                        query_embedding=query_embedding,
                                        limit=12,
                                    )
                                except Exception:
                                    results = []
                            elif probe:
                                results = await search_artifacts(db, probe, limit=12)
                            else:
                                results = []

                            if results:
                                for result in results:
                                    if str(result.get("scope", "")) != scope_id:
                                        continue
                                    if str(result.get("kind", "")) == "working_memory":
                                        continue
                                    artifact_id = str(result.get("id", ""))
                                    if not artifact_id:
                                        continue
                                    chain = await get_provenance_chain(db, artifact_id)
                                    check["search_scoped"] = True
                                    check["search_probe"] = probe
                                    check["search_hit_count"] = len(results)
                                    check["provenance_chain_len"] = len(chain)
                                    if chain:
                                        check["provenance_ok"] = True
                                        break
                                    check["provenance_check_target"] = artifact_id

                    rebuild_checks.append(check)

                def _scope_ok(entry: dict[str, Any]) -> bool:
                    if entry["entry_count"] <= 0:
                        return True
                    if not entry["scope_found"] or not entry["working_memory_present"]:
                        return False
                    if entry.get("working_memory_method") != "memory_repo":
                        return False
                    if not entry["search_scoped"] or not entry["provenance_ok"]:
                        return False
                    return True

                if probe_scopes:
                    rebuild_ok = all(_scope_ok(item) for item in rebuild_checks)
                else:
                    rebuild_ok = True

            finally:
                await db.close()

        result_payload = {
            "continuity_checks": continuity_checks,
            "continuity_ok": continuity_ok,
            "rebuild_run": not skip_rebuild,
            "rebuild_ok": rebuild_ok,
            "rebuild_result": db_rebuild_result,
            "scope_checks": rebuild_checks,
        }

        if json_output:
            print(json_mod.dumps(result_payload, default=str))
            return

        if not continuity_ok:
            console.print("[red]Continuity check failed[/red]")
            raise typer.Exit(code=1)

        if skip_rebuild:
            console.print("[green]Continuity check passed[/green]")
            console.print(f"[dim]Discovered durable scopes: {len(continuity_checks)}[/dim]")
            for entry in continuity_checks:
                console.print(
                    f"[dim]{entry['scope_name']}[/dim] entries={entry['entries']} "
                    f"canonical={entry['canonical_id']}"
                )
            return

        if rebuild_ok:
            console.print("[green]Rebuild proof passed[/green]")
            if db_rebuild_result:
                console.print(
                    f"[dim]Rebuild: mode={db_rebuild_result['mode']} "
                    f"runs={db_rebuild_result['runs_created']} "
                    f"turns={db_rebuild_result['turns_replayed']} "
                    f"wm={db_rebuild_result['working_memory_rehydrated']}[/dim]"
                )
            failed_search = [
                item
                for item in rebuild_checks
                if item.get("entry_count", 0) > 0 and not item.get("search_scoped")
            ]
            failed_provenance = [
                item
                for item in rebuild_checks
                if item.get("entry_count", 0) > 0 and not item.get("provenance_ok")
            ]
            if failed_search:
                console.print(f"[red]Scopes failing scoped search: {len(failed_search)}[/red]")
                for item in failed_search:
                    console.print(f"  - {item['scope_name']} ({item['scope_id']})")
                raise typer.Exit(code=1)
            if failed_provenance:
                console.print(
                    f"[yellow]Scopes with weak provenance checks: {len(failed_provenance)}[/yellow]"
                )
                for item in failed_provenance:
                    console.print(
                        f"  - {item['scope_name']} "
                        f"target={item.get('provenance_check_target', 'none')}"
                    )
                raise typer.Exit(code=1)
            validated = sum(1 for item in rebuild_checks if item.get("scope_id"))
            console.print(
                f"[dim]Scopes validated: {validated}[/dim]"
            )
        else:
            console.print("[red]Rebuild proof failed[/red]")
            raise typer.Exit(code=1)

    _run_async(_prove())


@admin_app.command("dashboard")
def dashboard(
    dev: bool = typer.Option(False, "--dev", help="Enable dev mode"),
) -> None:
    """Launch the TUI dashboard."""
    from .tui.app import run_dashboard

    run_dashboard()


# --- Daemon sub-commands ---


def _pid_alive(pid: int) -> bool:
    """Check if a process is running (Unix only).

    Uses signal 0 to probe. EPERM means the process exists but we lack
    permission (still alive). ESRCH means no such process (dead).
    """
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # Process exists but we can't signal it
    except ProcessLookupError:
        return False


@daemon_app.command("start")
def daemon_start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
) -> None:
    """Start the UC daemon (watcher + worker)."""
    from .daemon.core import run_daemon

    uc_home = get_uc_home()
    pid_file = uc_home / "daemon.pid"

    # Stale PID detection
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            pid = 0
        if pid and _pid_alive(pid):
            console.print(f"[yellow]Daemon already running (PID: {pid})[/yellow]")
            return
        # Stale PID file — clean up
        pid_file.unlink(missing_ok=True)

    if foreground:
        console.print("[bold]Starting UC daemon in foreground...[/bold]")
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
        try:
            _run_async(run_daemon())
        finally:
            pid_file.unlink(missing_ok=True)
    else:
        import subprocess
        import sys
        import time

        log_path = uc_home / "logs" / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")  # noqa: SIM115
        proc = subprocess.Popen(
            [sys.executable, "-m", "universal_context.cli", "daemon", "start", "-f"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        # Brief wait to confirm it started
        time.sleep(1)
        if proc.poll() is None:
            console.print(f"[green]Daemon started (PID: {proc.pid})[/green]")
            console.print(f"[dim]Log: {log_path}[/dim]")
        else:
            console.print("[red]Daemon failed to start. Check logs:[/red]")
            console.print(f"[dim]{log_path}[/dim]")


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the UC daemon."""
    uc_home = get_uc_home()
    pid_file = uc_home / "daemon.pid"
    if not pid_file.exists():
        console.print("[yellow]Daemon is not running.[/yellow]")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        console.print("[yellow]Invalid PID file — cleaned up.[/yellow]")
        return

    if not _pid_alive(pid):
        console.print(
            f"[yellow]Daemon process {pid} not found — cleaning up stale PID file.[/yellow]"
        )
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to daemon (PID: {pid})[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]Daemon process {pid} vanished — cleaning up PID file.[/yellow]")
    pid_file.unlink(missing_ok=True)


@daemon_app.command("status")
def daemon_status() -> None:
    """Show daemon status."""
    uc_home = get_uc_home()
    pid_file = uc_home / "daemon.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            console.print("[dim]Daemon is not running (invalid PID file).[/dim]")
            return
        if _pid_alive(pid):
            console.print(f"[green]Daemon running[/green] (PID: {pid})")
        else:
            console.print(f"[yellow]Daemon not running (stale PID: {pid})[/yellow]")
    else:
        console.print("[dim]Daemon is not running.[/dim]")


# --- Share sub-commands ---


@share_app.command("export")
def share_export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output path"),
    encrypt: bool = typer.Option(False, "--encrypt", "-e", help="Encrypt the bundle"),
    passphrase: str | None = typer.Option(None, "--passphrase", help="Encryption passphrase"),
) -> None:
    """Export a run as a portable share bundle (v2 with scope metadata)."""
    if encrypt and not passphrase:
        console.print("[red]--encrypt requires --passphrase[/red]")
        raise typer.Exit(code=1)

    async def _export():
        from .db.schema import apply_schema
        from .sharing.bundle import export_bundle

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            pw = passphrase if (encrypt or passphrase) else None
            result = await export_bundle(db, run_id, output_path=output, passphrase=pw)
            console.print(f"[green]Exported bundle:[/green] {result}")
        finally:
            await db.close()

    _run_async(_export())


@share_app.command("import")
def share_import(
    bundle: Path = typer.Argument(..., help="Path to share bundle"),
    project: Path | None = typer.Option(
        None, "--project", "-p", help="Import into this project's scope"
    ),
    passphrase: str | None = typer.Option(None, "--passphrase", help="Decryption passphrase"),
) -> None:
    """Import a share bundle into the local database."""

    async def _import():
        from .db.schema import apply_schema
        from .sharing.bundle import import_bundle

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            target_scope_id = None
            if project is not None:
                scope = await _resolve_scope(db, str(project.resolve()))
                if scope:
                    target_scope_id = str(scope["id"])

            result = await import_bundle(
                db,
                bundle,
                passphrase=passphrase,
                target_scope_id=target_scope_id,
            )
            console.print(
                f"[green]Imported:[/green] {result['turns_imported']} turns, "
                f"{result['artifacts_imported']} artifacts -> {result['run_id']}"
            )
            console.print(f"[dim]Scope: {result['scope_id']}[/dim]")
        finally:
            await db.close()

    _run_async(_import())


# --- Config sub-commands ---


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    from dataclasses import asdict

    import yaml
    from rich.syntax import Syntax

    from .config import UCConfig

    config = UCConfig.load()
    content = yaml.dump(asdict(config), default_flow_style=False)
    console.print(Syntax(content, "yaml", theme="monokai"))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Config value"),
) -> None:
    """Set a configuration value."""
    from dataclasses import fields

    from .config import UCConfig

    config = UCConfig.load()
    known = {f.name for f in fields(config)}
    if key not in known:
        console.print(f"[red]Unknown config key:[/red] {key}")
        console.print(f"[dim]Available keys: {', '.join(sorted(known))}[/dim]")
        raise typer.Exit(code=1)

    field_type = type(getattr(config, key))
    if field_type is bool:
        parsed = value.lower() in ("true", "1", "yes")
    elif field_type is int:
        parsed = int(value)
    elif field_type is float:
        parsed = float(value)
    else:
        parsed = value

    setattr(config, key, parsed)
    config.save()
    console.print(f"[green]Set[/green] {key} = {parsed}")


# --- Scope sub-commands ---


@scope_app.command("list")
def scope_list(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """List all scopes with stats (runs, turns, last activity)."""

    async def _list():
        from .db.queries import list_scopes_with_stats
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scopes = await list_scopes_with_stats(db)

            if json_output:
                print(json_mod.dumps([_sanitize_record(s) for s in scopes], default=str))
                return

            if not scopes:
                console.print("[dim]No scopes found.[/dim]")
                return

            table = Table(title="Scopes")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Canonical ID", max_width=40, style="dim")
            table.add_column("Path", max_width=40)
            table.add_column("Runs", justify="right")
            table.add_column("Turns", justify="right")
            table.add_column("Agents")
            table.add_column("Last Activity", style="dim")

            for s in scopes:
                agents = ", ".join(f"{k}({v})" for k, v in s.get("agent_breakdown", {}).items())
                last = str(s.get("last_activity", ""))[:19] if s.get("last_activity") else ""
                table.add_row(
                    str(s["id"]),
                    s["name"],
                    (s.get("canonical_id") or "")[:40],
                    (s.get("path") or "")[:40],
                    str(s.get("run_count", 0)),
                    str(s.get("turn_count", 0)),
                    agents,
                    last,
                )
            console.print(table)
        finally:
            await db.close()

    _run_async(_list())


@scope_app.command("show")
def scope_show(
    scope_ref: str = typer.Argument(..., help="Scope ID or name"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show details for a scope."""

    async def _show():
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scope = await _resolve_scope_ref(db, scope_ref)
            if not scope:
                console.print(f"[red]Scope not found:[/red] {scope_ref}")
                raise typer.Exit(code=1)

            if json_output:
                print(json_mod.dumps(_sanitize_record(scope), default=str))
                return

            console.print(f"[bold]{scope['id']}[/bold]")
            console.print(f"  Name: {scope['name']}")
            console.print(f"  Path: {scope.get('path', '(none)')}")
            console.print(f"  Canonical ID: {scope.get('canonical_id', '(none)')}")
            console.print(f"  Created: {scope.get('created_at', '')}")
        finally:
            await db.close()

    _run_async(_show())


@scope_app.command("update-path")
def scope_update_path(
    scope_ref: str = typer.Argument(..., help="Scope ID or name"),
    new_path: Path = typer.Argument(..., help="New filesystem path"),
) -> None:
    """Update a scope's filesystem path (e.g. when a project folder moves)."""

    async def _update():
        from .db.queries import find_scope_by_canonical_id, merge_scopes, update_scope
        from .db.schema import apply_schema
        from .git import resolve_canonical_id

        resolved = str(new_path.resolve())
        if not new_path.exists():
            console.print(f"[yellow]Warning: path does not exist on disk:[/yellow] {resolved}")

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scope = await _resolve_scope_ref(db, scope_ref)
            if not scope:
                console.print(f"[red]Scope not found:[/red] {scope_ref}")
                raise typer.Exit(code=1)

            sid = str(scope["id"])
            canonical_id = resolve_canonical_id(new_path.resolve())

            # Check if another scope already has this canonical_id
            existing = await find_scope_by_canonical_id(db, canonical_id)
            if existing and str(existing["id"]) != sid:
                # Auto-merge into existing scope (same project, different paths)
                existing_id = str(existing["id"])
                await merge_scopes(db, sid, existing_id)
                # Update the surviving scope's path to the new location
                await update_scope(db, existing_id, path=resolved)
                console.print(
                    f"[green]Merged[/green] {sid} into {existing_id} "
                    f"(shared canonical ID: {canonical_id})"
                )
                console.print(f"[dim]Updated path on {existing_id} -> {resolved}[/dim]")
            else:
                await update_scope(db, sid, path=resolved, canonical_id=canonical_id)
                console.print(f"[green]Updated[/green] {sid} path -> {resolved}")
                console.print(f"[dim]Canonical ID: {canonical_id}[/dim]")
        finally:
            await db.close()

    _run_async(_update())


@scope_app.command("rename")
def scope_rename(
    scope_ref: str = typer.Argument(..., help="Scope ID or name"),
    new_name: str = typer.Argument(..., help="New display name"),
) -> None:
    """Change a scope's display name."""

    async def _rename():
        from .db.queries import update_scope
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scope = await _resolve_scope_ref(db, scope_ref)
            if not scope:
                console.print(f"[red]Scope not found:[/red] {scope_ref}")
                raise typer.Exit(code=1)

            sid = str(scope["id"])
            await update_scope(db, sid, name=new_name)
            console.print(f"[green]Renamed[/green] {sid} -> {new_name}")
        finally:
            await db.close()

    _run_async(_rename())


@scope_app.command("merge")
def scope_merge(
    source: str = typer.Argument(..., help="Source scope ID or name"),
    into: str = typer.Option(..., "--into", help="Target scope ID or name"),
) -> None:
    """Move all runs from source scope into target, then delete source."""

    async def _merge():
        from .db.queries import merge_scopes
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            src = await _resolve_scope_ref(db, source)
            tgt = await _resolve_scope_ref(db, into)
            if not src:
                console.print(f"[red]Source scope not found:[/red] {source}")
                raise typer.Exit(code=1)
            if not tgt:
                console.print(f"[red]Target scope not found:[/red] {into}")
                raise typer.Exit(code=1)

            src_id = str(src["id"])
            tgt_id = str(tgt["id"])
            if src_id == tgt_id:
                console.print("[red]Source and target are the same scope.[/red]")
                raise typer.Exit(code=1)

            await merge_scopes(db, src_id, tgt_id)
            console.print(
                f"[green]Merged[/green] {src_id} ({src['name']}) -> {tgt_id} ({tgt['name']})"
            )
        finally:
            await db.close()

    _run_async(_merge())


@scope_app.command("rm")
def scope_rm(
    scope_ref: str = typer.Argument(..., help="Scope ID or name"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation"),
) -> None:
    """Delete a scope and all its runs/turns/artifacts."""

    async def _rm():
        from .db.queries import delete_scope
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scope = await _resolve_scope_ref(db, scope_ref)
            if not scope:
                console.print(f"[red]Scope not found:[/red] {scope_ref}")
                raise typer.Exit(code=1)

            sid = str(scope["id"])
            if not confirm:
                console.print(
                    f"[yellow]This will delete scope {sid} ({scope['name']}) "
                    f"and ALL its data.[/yellow]"
                )
                console.print("Use --confirm to proceed.")
                raise typer.Exit(code=1)

            await delete_scope(db, sid)
            console.print(f"[green]Deleted[/green] {sid} ({scope['name']})")
        finally:
            await db.close()

    _run_async(_rm())


@scope_app.command("cleanup")
def scope_cleanup(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed"),
) -> None:
    """Find and remove garbage scopes (date folders, hash dirs, empty scopes)."""

    async def _cleanup():
        import re

        from .db.queries import delete_scope, list_scopes_with_stats
        from .db.schema import apply_schema

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scopes = await list_scopes_with_stats(db)

            garbage: list[dict[str, Any]] = []
            for s in scopes:
                path = s.get("path", "") or ""
                name = s.get("name", "") or ""
                canonical_id = s.get("canonical_id") or ""
                is_garbage = False
                reason = ""

                # Skip scopes with a valid git-backed canonical_id
                if canonical_id and not canonical_id.startswith("path://"):
                    continue

                # Date-like directories (e.g. "03", "09", "2025")
                if re.match(r"^\d{1,4}$", name):
                    is_garbage = True
                    reason = "date-like directory name"

                # SHA256 hash directories (64 hex chars)
                elif re.match(r"^[0-9a-f]{40,64}$", name):
                    is_garbage = True
                    reason = "hash directory name"

                # Paths inside .codex/sessions or .gemini/tmp
                elif "/.codex/sessions/" in path or "/.gemini/tmp/" in path:
                    is_garbage = True
                    reason = "session metadata path"

                # Empty scopes with no runs
                elif s.get("run_count", 0) == 0:
                    is_garbage = True
                    reason = "empty scope (no runs)"

                if is_garbage:
                    garbage.append({**s, "reason": reason})

            if not garbage:
                console.print("[green]No garbage scopes found.[/green]")
                return

            table = Table(title="Garbage Scopes" + (" (dry run)" if dry_run else ""))
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Path", max_width=40)
            table.add_column("Runs", justify="right")
            table.add_column("Reason", style="yellow")

            for g in garbage:
                table.add_row(
                    str(g["id"]),
                    g["name"],
                    (g.get("path") or "")[:40],
                    str(g.get("run_count", 0)),
                    g["reason"],
                )
            console.print(table)

            if dry_run:
                console.print(
                    f"\n[dim]{len(garbage)} scopes would be removed. "
                    f"Run without --dry-run to delete.[/dim]"
                )
            else:
                for g in garbage:
                    await delete_scope(db, str(g["id"]))
                console.print(f"\n[green]Removed {len(garbage)} garbage scopes.[/green]")
        finally:
            await db.close()

    _run_async(_cleanup())


# --- Memory sub-commands ---


@memory_app.command("show")
def memory_show(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    format_mode: str = typer.Option(
        "plain",
        "--format",
        "-f",
        help="Output format: plain, inject",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show the working memory for a project."""

    async def _show():
        from .git import resolve_canonical_id
        from .memory_repo import (
            MEMORY_SECTIONS,
            list_scope_sections,
            render_section_text,
        )

        project_path = str((project or Path(".")).resolve())
        project_name = Path(project_path).name
        scope_payload: dict[str, Any] = {"name": project_name, "path": project_path}
        memory_payload: dict[str, Any] | None = None
        content = ""

        canonical_id = ""
        try:
            canonical_id = resolve_canonical_id(Path(project_path))
        except Exception:
            canonical_id = ""

        if canonical_id:
            repo_sections = list_scope_sections(
                canonical_id=canonical_id,
                display_name=project_name,
                scope_path=project_path,
            )
            repo_parts = []
            for section in MEMORY_SECTIONS:
                entries = repo_sections.get(section, [])
                rendered = render_section_text(entries)
                if rendered:
                    title = section.replace("_", " ").title()
                    repo_parts.append(f"# {title}\n\n{rendered}")
            if repo_parts:
                content = "\n\n".join(repo_parts)
                scope_payload["canonical_id"] = canonical_id
                memory_payload = {
                    "id": f"memory_repo:{canonical_id}",
                    "content": content,
                    "created_at": "",
                    "metadata": {"method": "memory_repo"},
                }

        if not content:
            from .db.queries import get_working_memory
            from .db.schema import apply_schema

            db = _get_db()
            await db.connect()
            try:
                await apply_schema(db)
                scope = await _resolve_scope(db, project_path)
                if not scope:
                    if format_mode == "inject":
                        return  # Silent for hook injection
                    if json_output:
                        print(json_mod.dumps({"error": "no_scope", "project": project_path}))
                    else:
                        console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                    return

                scope_payload = _sanitize_record(scope)
                scope_id = str(scope["id"])
                memory = await get_working_memory(db, scope_id)
                if not memory:
                    if format_mode == "inject":
                        return  # Silent for hook injection
                    if json_output:
                        print(
                            json_mod.dumps(
                                {
                                    "error": "no_memory",
                                    "scope": _sanitize_record(scope),
                                }
                            )
                        )
                    else:
                        console.print("[dim]No working memory for this project yet.[/dim]")
                        console.print("[dim]Run: uc memory refresh --project .[/dim]")
                    return
                content = memory.get("content", "")
                memory_payload = memory
            finally:
                await db.close()

        if json_output:
            print(
                json_mod.dumps(
                    {
                        **_sanitize_record(memory_payload or {}),
                        "scope": scope_payload,
                    },
                    default=str,
                )
            )
        elif format_mode == "inject":
            # Raw markdown for hook injection — no Rich formatting
            print(content)
        else:
            from rich.markdown import Markdown
            from rich.panel import Panel

            console.print(
                Panel(
                    Markdown(content),
                    title=f"Working Memory: {scope_payload.get('name', '')}",
                    border_style="blue",
                )
            )
            created = str(memory_payload.get("created_at", ""))[:19] if memory_payload else ""
            method = memory_payload.get("metadata", {}).get("method", "") if memory_payload else ""
            memory_id = memory_payload.get("id", "") if memory_payload else ""
            console.print(
                f"[dim]Updated: {created}  Method: {method}  ID: {memory_id}[/dim]"
            )

    _run_async(_show())


@memory_app.command("refresh")
def memory_refresh(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Force-regenerate the working memory now (runs inline, no daemon needed)."""

    async def _refresh():
        from .config import UCConfig
        from .db.queries import (
            backfill_artifact_scopes,
            backfill_canonical_ids,
            get_working_memory,
        )
        from .db.schema import apply_schema, rebuild_hnsw_index
        from .embed import create_embed_provider
        from .llm import create_llm_fn

        project_path = str((project or Path(".")).resolve())
        config = UCConfig.load()

        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            # Backfill canonical_ids on scopes (git-aware dedup)
            cid_backfilled = await backfill_canonical_ids(db)
            if cid_backfilled and not json_output:
                console.print(f"[dim]Backfilled canonical_id on {cid_backfilled} scopes.[/dim]")

            # Backfill scope on any existing artifacts that lack it
            backfilled = await backfill_artifact_scopes(db)
            if backfilled and not json_output:
                console.print(f"[dim]Backfilled scope on {backfilled} artifacts.[/dim]")

            scope = await _resolve_scope(db, project_path)
            if not scope:
                if json_output:
                    print(json_mod.dumps({"error": "no_scope", "project": project_path}))
                else:
                    console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                return

            scope_id = str(scope["id"])
            llm_fn = await create_llm_fn(config)
            if llm_fn is None:
                if json_output:
                    print(json_mod.dumps({"error": "no_llm", "message": "LLM not configured"}))
                else:
                    console.print("[red]LLM not configured. Working memory requires an LLM.[/red]")
                return

            embed_provider = await create_embed_provider(config)

            from .daemon.processors.memory import WorkingMemoryProcessor

            processor = WorkingMemoryProcessor(
                llm_fn=llm_fn,
                embed_fn=embed_provider,
                max_summaries=config.memory_max_summaries,
            )

            if not json_output:
                console.print(f"[bold]Refreshing working memory for {scope.get('name')}...[/bold]")

            job = {"target": scope_id}
            result = await processor.process(db, job)

            # Rebuild HNSW index so new embeddings are searchable
            rebuilt = await rebuild_hnsw_index(db)

            if json_output:
                # Include the actual memory content
                memory = await get_working_memory(db, scope_id)
                print(
                    json_mod.dumps(
                        {
                            "result": result,
                            "memory": _sanitize_record(memory) if memory else None,
                            "hnsw_rebuilt": rebuilt,
                            "backfilled": backfilled,
                        },
                        default=str,
                    )
                )
            else:
                if result.get("status") == "skipped":
                    console.print(f"[yellow]Skipped:[/yellow] {result.get('reason')}")
                else:
                    console.print(
                        f"[green]Working memory updated[/green] "
                        f"({result.get('summaries_used', 0)} summaries distilled)"
                    )
                    if rebuilt:
                        console.print("[dim]HNSW index rebuilt.[/dim]")
        finally:
            await db.close()

    _run_async(_refresh())


def _split_legacy_working_memory(raw: str) -> list[tuple[str, str]]:
    import re

    title_re = re.compile(r"^##\s+(?P<title>.+)$")

    def classify(title: str) -> str:
        lowered = title.lower()
        if "architecture" in lowered:
            return "architecture"
        if "gotcha" in lowered or "learned" in lowered:
            return "state"
        if "current" in lowered or "recent" in lowered or "active" in lowered:
            return "state"
        return "state"

    current_title = "state"
    current_body: list[str] = []
    blocks: list[tuple[str, str]] = []

    for line in raw.splitlines():
        m = title_re.match(line.strip())
        if not m:
            current_body.append(line)
            continue

        if any(token.strip() for token in current_body):
            blocks.append((classify(current_title), "\n".join(current_body).strip()))
            current_body = []
        current_title = m.group("title")

    if any(token.strip() for token in current_body):
        blocks.append((classify(current_title), "\n".join(current_body).strip()))
    return blocks


@memory_app.command("sync")
def memory_sync(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    target: str = typer.Option(
        "AGENTS.md",
        "--target",
        "-t",
        help="Target file to inject into after refresh",
    ),
) -> None:
    """Refresh project memory, then inject it into a target file."""
    memory_refresh(project=project, json_output=False)
    memory_inject(project=project, target=target)


@memory_app.command("migrate-db")
def memory_migrate_db(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Migrate one scope (default: cwd)",
    ),
    all_scopes: bool = typer.Option(
        False,
        "--all",
        help="Migrate all scopes from DB.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Migrate existing DB working-memory into canonical memory files."""

    async def _migrate():
        import hashlib
        from pathlib import Path as _Path

        from .db.queries import get_working_memory, list_scopes
        from .db.schema import apply_schema
        from .git import resolve_canonical_id
        from .memory_repo import (
            MEMORY_SECTIONS,
            append_section_entry,
            get_memory_migrations_root,
            read_section_entries,
        )

        project_path = (project or _Path(".")).resolve()
        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            if all_scopes:
                target_scopes = await list_scopes(db)
            else:
                scope = await _resolve_scope(db, str(project_path))
                if not scope:
                    if json_output:
                        print(json_mod.dumps({"error": "no_scope", "project": str(project_path)}))
                    else:
                        console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                    return
                target_scopes = [scope]

            migrated: list[str] = []
            skipped: list[str] = []
            failed: list[dict[str, str]] = []
            migration_root = get_memory_migrations_root()
            migration_root.mkdir(parents=True, exist_ok=True)

            for scope in target_scopes:
                scope_id = str(scope["id"])
                memory = await get_working_memory(db, scope_id)
                if not memory:
                    skipped.append(scope_id)
                    continue

                scope_name = str(scope.get("name") or "project")
                scope_path = str(scope.get("path") or project_path)
                canonical_id = scope.get("canonical_id") or resolve_canonical_id(_Path(scope_path))
                if not canonical_id:
                    failed.append(
                        {"scope_id": scope_id, "reason": "missing canonical_id"}
                    )
                    continue

                marker_id = hashlib.sha1(str(canonical_id).encode("utf-8")).hexdigest()[:16]
                marker = migration_root / f"{marker_id}.json"
                if marker.exists():
                    skipped.append(scope_id)
                    continue

                content = memory.get("content", "")
                if not isinstance(content, str):
                    content = str(content or "")
                if not content.strip():
                    failed.append(
                        {
                            "scope_id": scope_id,
                            "canonical_id": str(canonical_id),
                            "reason": "empty working memory",
                        }
                    )
                    continue

                blocks = _split_legacy_working_memory(content)
                wrote = False
                for section, block in blocks:
                    if section not in MEMORY_SECTIONS:
                        section = "state"
                    if not block.strip():
                        continue
                    existing = read_section_entries(
                        canonical_id=canonical_id,
                        section=section,
                        display_name=scope_name,
                        scope_path=scope_path,
                    )
                    duplicate = any(
                        e.get("source") == "migrated_db"
                        and e.get("content", "").strip() == block.strip()
                        and str(e.get("scope_canonical_id")) == canonical_id
                        for e in existing
                    )
                    if duplicate:
                        continue
                    append_section_entry(
                        canonical_id=canonical_id,
                        section=section,
                        display_name=scope_name,
                        content=block.strip(),
                        memory_type="durable_fact",
                        confidence=0.8,
                        manual=False,
                        source="migrated_db",
                        scope_path=scope_path,
                    )
                    wrote = True

                if wrote:
                    marker.write_text(
                        json_mod.dumps(
                            {"scope_id": scope_id, "canonical_id": canonical_id},
                        ),
                        encoding="utf-8",
                    )
                    migrated.append(scope_id)
                else:
                    skipped.append(scope_id)

            if json_output:
                print(
                    json_mod.dumps(
                        {
                            "migrated": migrated,
                            "skipped": skipped,
                            "failed": failed,
                            "ok": not failed,
                        }
                    )
                )
            else:
                if migrated:
                    console.print(f"[green]Migrated scopes:[/green] {len(migrated)}")
                if skipped:
                    console.print(f"[dim]Skipped scopes:[/dim] {len(skipped)}")
                if failed:
                    console.print(f"[red]Failed migrations:[/red] {len(failed)}")
                    for item in failed:
                        console.print(f"  - {item['scope_id']}: {item['reason']}")
            if failed:
                raise typer.Exit(code=1)
        finally:
            await db.close()

    _run_async(_migrate())


def _infer_remember_type(section: str, text: str) -> str:
    """Fast deterministic memory-type inference for `uc remember`."""
    normalized = text.lower()
    if section == "procedures":
        return "procedure"
    if section == "preferences":
        return "decision"
    if section == "open_questions":
        return "open_question"
    if "why" in normalized or "decision" in normalized or "chose" in normalized:
        return "decision"
    if "how" in normalized and "do" in normalized:
        return "procedure"
    return "durable_fact"


def _infer_remember_section(section: str, memory_type: str) -> str:
    """Derive section when explicit section inference is requested."""
    inferred_type = memory_type if memory_type else "durable_fact"
    if section not in ("auto",):
        return section

    if inferred_type == "procedure":
        return "procedures"
    if inferred_type == "decision":
        return "preferences"
    if inferred_type == "open_question":
        return "open_questions"
    return "state"


def _coerce_evidence_payload(payload: Any) -> list[dict[str, str]]:
    """Convert one evidence payload into canonical evidence rows."""
    if isinstance(payload, dict):
        normalized: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            k = key.strip().lower()
            if k in {"artifact", "artifact_id", "artifactid"}:
                k = "artifact_id"
            elif k in {"run", "run_id"}:
                k = "run_id"
            elif k in {"turn", "turn_id"}:
                k = "turn_id"
            elif k in {"session", "session_id"}:
                k = "session_id"
            else:
                continue

            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if not value:
                continue
            normalized[k] = value
        return [normalized] if normalized else []

    if isinstance(payload, (int, float, str)):
        text = str(payload).strip()
        if not text:
            return []
        if ":" in text:
            key, value = [part.strip() for part in text.split(":", 1)]
            if not value:
                return []
            if key in {"artifact", "artifact_id", "artifactid"}:
                return [{"artifact_id": value}]
            if key in {"run", "run_id"}:
                return [{"run_id": value}]
            if key in {"turn", "turn_id"}:
                return [{"turn_id": value}]
            if key in {"session", "session_id"}:
                return [{"session_id": value}]

        return [{"artifact_id": text}]

    if isinstance(payload, list):
        rows: list[dict[str, str]] = []
        for item in payload:
            rows.extend(_coerce_evidence_payload(item))
        return rows

    return []


def _parse_remember_evidence(items: list[str] | None) -> list[dict[str, str]]:
    """Parse evidence input as JSON, CSV, or repeated legacy IDs."""
    import csv

    if not items:
        return []

    rows: list[dict[str, str]] = []
    for raw in items:
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue

        parsed: Any | None = None
        try:
            parsed = json_mod.loads(text)
        except json_mod.JSONDecodeError:
            parsed = None

        if parsed is not None:
            rows.extend(_coerce_evidence_payload(parsed))
            continue

        for row in csv.reader([text], skipinitialspace=True):
            for token in row:
                token = token.strip()
                if not token:
                    continue
                rows.extend(_coerce_evidence_payload(token))

    dedup: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row:
            continue
        dedup[json_mod.dumps(row, sort_keys=True)] = row
    return list(dedup.values())


def _normalize_remember_confidence(value: Any, fallback: float = 0.85) -> float:
    """Clamp confidence into [0.0, 1.0] and normalize fallback on bad values."""
    try:
        value_num = float(value)
    except (TypeError, ValueError):
        return fallback

    if value_num < 0.0:
        return 0.0
    if value_num > 1.0:
        return 1.0
    return value_num


def _normalize_skill_title(content: str) -> str:
    """Build a human title for a promoted skill from memory content."""
    first_line = content.strip().splitlines()[0].strip("- ").strip()
    if not first_line:
        return "Promoted Procedure"

    title = first_line.replace("#", "").strip()
    if len(title) <= 80:
        return title
    return f"{title[:77]}..."


def _render_provenance_rows(entries: list[dict[str, Any]]) -> list[str]:
    """Normalize and stringify evidence rows into readable bullets."""
    rendered: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parts: list[str] = []
        for key in ("artifact_id", "run_id", "turn_id", "session_id"):
            value = entry.get(key)
            if value:
                parts.append(f"{key}:{value}")
        if parts:
            rendered.append(" - " + ", ".join(parts))
    if not rendered:
        return ["- No provenance identifiers present."]
    return rendered


@app.command("remember")
@memory_app.command("remember")
def memory_remember(
    content: str = typer.Argument(..., help="Memory payload"),
    project: Path = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project path (required).",
    ),
    section: str = typer.Option(
        "auto",
        "--section",
        "-s",
        help=(
            "Section: architecture, state, procedures, preferences, open_questions. "
            "Use auto to infer section from type."
        ),
    ),
    memory_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help=(
            "Entry type (required): durable_fact, procedure, decision, open_question."
        ),
    ),
    confidence: float = typer.Option(
        0.85,
        "--confidence",
        min=0.0,
        max=1.0,
        help="Confidence score 0.0-1.0.",
    ),
    produced_by_model: str = typer.Option(
        "manual",
        "--produced-by-model",
        help="Source model or actor.",
    ),
    source: str = typer.Option(
        "remember",
        "--source",
        help="Evidence source label (distilled, remember, migrated_db, import).",
    ),
    evidence: list[str] = typer.Option(
        None,
        "--evidence",
        help="Evidence IDs (e.g. artifact:abc123). Can be repeated.",
    ),
) -> None:
    """Append a deterministic memory entry directly into durable memory files."""
    from .git import resolve_canonical_id
    from .memory_repo import MEMORY_SECTIONS, append_section_entry

    normalized_section = section.strip().lower() or "auto"
    if normalized_section != "auto" and normalized_section not in MEMORY_SECTIONS:
        raise typer.BadParameter(
            f"Invalid section '{section}'. Expected one of {', '.join(MEMORY_SECTIONS)} or auto."
        )

    normalized_type = memory_type.strip().lower() if isinstance(memory_type, str) else ""
    allowed_types = ("durable_fact", "procedure", "decision", "open_question")
    if normalized_type not in allowed_types:
        raise typer.BadParameter(
            f"Invalid --type '{memory_type}'. Expected one of {', '.join(allowed_types)}."
        )
    memory_type = normalized_type

    resolved_section = _infer_remember_section(normalized_section, memory_type)
    if normalized_section == "auto" and resolved_section != normalized_section:
        section = resolved_section
    else:
        section = normalized_section

    scope_path = project.resolve()
    canonical_id = resolve_canonical_id(scope_path)
    if not canonical_id:
        raise typer.BadParameter(f"Unable to resolve canonical scope id for project: {scope_path}")
    display_name = scope_path.name

    parsed_evidence = _parse_remember_evidence(evidence)
    normalized_confidence = _normalize_remember_confidence(confidence)

    append_section_entry(
        canonical_id=canonical_id,
        section=section,
        display_name=display_name,
        content=content,
        memory_type=memory_type,
        confidence=normalized_confidence,
        manual=True,
        source=source.strip() or "remember",
        produced_by_model=produced_by_model,
        evidence=parsed_evidence,
        scope_path=str(scope_path),
    )

    console.print("[green]Remembered entry written to durable memory[/green]")


def _normalize_repo_target(project: Path | None) -> str | None:
    """Normalize a project path to the canonical-id form used by memory scope registry."""
    if project is None:
        return None
    from .git import resolve_canonical_id
    from .memory_repo import normalize_canonical_id

    return normalize_canonical_id(resolve_canonical_id(project))


def _collect_memory_repo_scopes(
    project: Path | None = None,
) -> list[dict[str, Any]]:
    """Load memory scope records from scope-map, optionally filtered by project."""
    from .memory_repo import list_scope_registry, normalize_canonical_id

    records = list_scope_registry()
    if not records:
        return []

    if project is None:
        return [record.__dict__ for record in records]

    target = _normalize_repo_target(project)
    if not target:
        return []

    project_path = str(project.resolve())
    matched: list[dict[str, Any]] = []
    for record in records:
        candidates = [record.canonical_id, record.canonical_url, record.remote_url, record.path]
        if any(normalize_canonical_id(candidate or "") == target for candidate in candidates):
            matched.append(record.__dict__)
            continue
        if record.path == project_path:
            matched.append(record.__dict__)
    return matched


def _gather_scope_entries(
    scope_record: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], int]:
    """Load all section entries for a scoped memory repo row."""
    from .memory_repo import list_scope_sections

    canonical_id = scope_record["canonical_id"]
    display_name = str(
        scope_record.get("display_name")
        or scope_record.get("scope_name")
        or "project"
    )
    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name=display_name,
        scope_path=scope_record.get("path"),
    )
    flat_entries = [entry for entries in sections.values() for entry in entries]
    total = sum(len(entries) for entries in sections.values())
    return sections, flat_entries, total


def _derive_probe_text(entries: list[dict[str, Any]]) -> str:
    """Derive a short probe phrase from the first durable memory entry."""
    import re

    for entry in entries:
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_/#.:+-]+", content)]
        if len(tokens) >= 2:
            return " ".join(tokens[:6])
        return content[:60]
    return ""


def _build_scope_index(scope_records: list[dict[str, Any]]) -> dict[str, str]:
    """Build canonical-id lookup for DB scopes."""
    from .memory_repo import normalize_canonical_id

    index: dict[str, str] = {}
    for scope in scope_records:
        scope_id = str(scope.get("id", ""))
        if not scope_id:
            continue

        canonical_id = normalize_canonical_id(
            str(scope.get("canonical_id") or scope.get("path") or "")
        )
        if canonical_id:
            index[canonical_id] = scope_id

    return index


@admin_app.command("promote-skills")
def promote_skills(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Single project scope to inspect (default: all)",
    ),
    min_occurrences: int = typer.Option(
        2,
        "--min-occurrences",
        help="Minimum repeated mentions before promoting a procedure.",
    ),
    min_confidence: float = typer.Option(
        0.8,
        "--min-confidence",
        min=0.0,
        max=1.0,
        help="Minimum average confidence for promotion candidates.",
    ),
    min_evidence: int = typer.Option(
        1,
        "--min-evidence",
        help="Minimum evidence rows across matching entries.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run only; do not write files."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing skill files."),
) -> None:
    """Promote stable procedural memories into durable skill artifacts."""
    import hashlib
    import re

    from .db.queries import list_scopes
    from .db.schema import apply_schema
    from .git import resolve_canonical_id
    from .memory_repo import get_memory_skills_root, list_scope_sections, slugify_name

    normalized_min_conf = _normalize_remember_confidence(min_confidence, fallback=0.8)

    async def _promote() -> None:
        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)

            if project:
                scope = await _resolve_scope(db, str((project).resolve()))
                scopes = [scope] if scope else []
            else:
                scopes = await list_scopes(db)

            total_candidates = 0
            total_promoted = 0
            total_skipped = 0

            for scope in scopes:
                if not scope:
                    continue
                scope_id = str(scope.get("id", ""))
                scope_name = str(scope.get("name", "project"))
                scope_path = scope.get("path")
                canonical_id = scope.get("canonical_id")
                if not canonical_id and scope_path:
                    canonical_id = resolve_canonical_id(Path(scope_path))
                if not canonical_id:
                    total_skipped += 1
                    continue

                sections = list_scope_sections(
                    canonical_id=canonical_id,
                    display_name=scope_name,
                    scope_path=scope_path or str((project or Path(".")).resolve()),
                )
                entries = sections.get("procedures", [])
                if not entries:
                    continue

                grouped: dict[str, dict[str, Any]] = {}
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    content = str(entry.get("content", "")).strip()
                    if not content:
                        continue
                    memory_type = entry.get("type", "durable_fact")
                    if memory_type != "procedure":
                        continue

                    conf = _normalize_remember_confidence(entry.get("confidence"), fallback=0.0)
                    evidence = entry.get("evidence", [])

                    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    bucket = grouped.setdefault(
                        key,
                        {
                            "content": content,
                            "count": 0,
                            "confidence_total": 0.0,
                            "max_confidence": 0.0,
                            "evidence": [],
                            "sources": [],
                        },
                    )
                    bucket["count"] += 1
                    bucket["confidence_total"] += conf
                    if conf > bucket["max_confidence"]:
                        bucket["max_confidence"] = conf
                    if evidence:
                        bucket["evidence"].extend(
                            item for item in evidence if isinstance(item, dict)
                        )
                    bucket["sources"].append(
                        {
                            "id": str(entry.get("entry_id", "")),
                            "scope_id": scope_id,
                            "scope_canonical_id": str(entry.get("scope_canonical_id", "")),
                            "produced_by_model": str(entry.get("produced_by_model", "")),
                            "manual": bool(entry.get("manual", False)),
                        }
                    )

                for bucket in grouped.values():
                    count = bucket["count"]
                    avg_confidence = (
                        bucket["confidence_total"] / count if count else 0.0
                    )
                    evidence_count = len(bucket["evidence"])
                    if (
                        count < min_occurrences
                        or avg_confidence < normalized_min_conf
                        or evidence_count < min_evidence
                    ):
                        continue

                    total_candidates += 1
                    title = _normalize_skill_title(bucket["content"])
                    title_hash = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
                    slug = f"{slugify_name(title)}--{title_hash}"
                    skill_dir = get_memory_skills_root() / slug
                    skill_path = skill_dir / "SKILL.md"
                    metadata_path = skill_dir / "metadata.yaml"

                    if dry_run:
                        console.print(
                            f"[yellow]DRY[/yellow] {scope_name} -> {title} "
                            f"(x{count}, conf={avg_confidence:.2f})"
                        )
                        continue

                    if skill_path.exists() and not overwrite:
                        total_skipped += 1
                        continue

                    skill_dir.mkdir(parents=True, exist_ok=True)
                    provenance_lines = _render_provenance_rows(bucket["evidence"])
                    metadata = {
                        "type": "procedure",
                        "name": title,
                        "scope_id": scope_id,
                        "scope_name": scope_name,
                        "scope_canonical_id": canonical_id,
                        "scope_path": scope_path,
                        "occurrences": count,
                        "max_confidence": bucket["max_confidence"],
                        "avg_confidence": avg_confidence,
                        "min_confidence": normalized_min_conf,
                        "min_occurrences": min_occurrences,
                        "min_evidence": min_evidence,
                        "source_count": len(bucket["sources"]),
                    }

                    skill_steps = "\n".join(
                        line
                        for line in re.split(r"\n+", str(bucket["content"]).strip())
                        if line.strip()
                    )
                    if skill_steps and not skill_steps.startswith("-"):
                        skill_steps = "\n".join(
                            f"- {line.strip()}" for line in skill_steps.splitlines() if line.strip()
                        )

                    skill_doc = (
                        f"# {title}\n\n"
                        f"- Scope: {scope_name}\n"
                        f"- Canonical ID: {canonical_id}\n"
                        f"- Occurrences: {count}\n"
                        f"- Average confidence: {avg_confidence:.2f}\n"
                        "- Version: 1\n\n"
                        "## Preconditions\n\n"
                        "- Validate runtime assumptions before running.\n\n"
                        "## Steps\n\n"
                        f"{skill_steps or '- Unknown (derive from procedure memory entries).'}\n\n"
                        "## Validation\n\n"
                        "- Confirm expected outcome and rollback criteria.\n\n"
                        "## Failure Modes\n\n"
                        "- Procedure drift due to changed architecture.\n"
                        "- Missing inputs or invalid preconditions.\n"
                        "- Tooling/API behavior changed since capture.\n\n"
                        "## Provenance\n\n"
                        + "\n".join(provenance_lines)
                        + "\n"
                    )

                    skill_path.write_text(skill_doc, encoding="utf-8")
                    import yaml

                    metadata_path.write_text(
                        yaml.safe_dump(metadata, sort_keys=False),
                        encoding="utf-8",
                    )
                    total_promoted += 1
                    console.print(f"[green]Promoted skill:[/green] {scope_name} / {title}")

            console.print(f"[green]Skill promotion complete[/green]: {total_promoted} promoted")
            if dry_run:
                console.print(f"[dim]Candidates evaluated: {total_candidates}[/dim]")
            if total_skipped:
                console.print(f"[dim]Skipped existing: {total_skipped}[/dim]")
            if total_promoted == 0:
                console.print("[dim]No procedures met promotion thresholds.[/dim]")
            if total_candidates == 0:
                console.print("[dim]No promotion candidates found.[/dim]")
        finally:
            await db.close()

    _run_async(_promote())


@memory_app.command("history")
def memory_history(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Max versions to show"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show previous versions of the working memory."""

    async def _history():
        from .db.queries import get_working_memory_history
        from .db.schema import apply_schema

        project_path = str((project or Path(".")).resolve())
        db = _get_db()
        await db.connect()
        try:
            await apply_schema(db)
            scope = await _resolve_scope(db, project_path)
            if not scope:
                if json_output:
                    print(json_mod.dumps({"error": "no_scope", "project": project_path}))
                else:
                    console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                return

            scope_id = str(scope["id"])
            versions = await get_working_memory_history(db, scope_id, limit=limit)

            if json_output:
                print(
                    json_mod.dumps(
                        [_sanitize_record(v) for v in versions],
                        default=str,
                    )
                )
                return

            if not versions:
                console.print("[dim]No working memory versions found.[/dim]")
                return

            table = Table(title=f"Working Memory History: {scope.get('name', '')}")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("ID", style="dim")
            table.add_column("Method")
            table.add_column("Created", style="dim")
            table.add_column("Preview", max_width=50)

            for i, v in enumerate(versions, 1):
                method = v.get("metadata", {}).get("method", "")
                created = str(v.get("created_at", ""))[:19]
                preview = (v.get("content", "") or "")[:50].replace("\n", " ")
                table.add_row(
                    str(i),
                    str(v.get("id", "")),
                    method,
                    created,
                    preview,
                )
            console.print(table)
        finally:
            await db.close()

    _run_async(_history())


@memory_app.command("install-hook")
def memory_install_hook() -> None:
    """Install the SessionStart hook into Claude Code settings.

    Adds a hook that runs `uc memory show --project . --format inject`
    at the start of each Claude Code session, injecting project working
    memory into the agent's context automatically.
    """
    settings_path = Path.home() / ".claude" / "settings.json"

    hook_command = "uc memory show --project . --format inject 2>/dev/null || true"

    if settings_path.exists():
        settings = json_mod.loads(settings_path.read_text(encoding="utf-8"))
    else:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = {}

    hooks = settings.setdefault("hooks", {})
    session_start = hooks.setdefault("SessionStart", [])

    # Check if already installed
    for entry in session_start:
        for h in entry.get("hooks", []):
            if h.get("command", "") == hook_command:
                console.print("[dim]Hook already installed.[/dim]")
                return

    session_start.append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": hook_command,
                }
            ],
        }
    )

    settings_path.write_text(
        json_mod.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Installed SessionStart hook[/green] in {settings_path}")
    console.print(f"[dim]Command: {hook_command}[/dim]")


# Sentinel markers for AGENTS.md injection
_MEMORY_START = "<!-- UC:MEMORY:START -->"
_MEMORY_END = "<!-- UC:MEMORY:END -->"


@memory_app.command("inject")
def memory_inject(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    target: str = typer.Option(
        "AGENTS.md",
        "--target",
        "-t",
        help="Target file to inject into (AGENTS.md, CLAUDE.md, etc.)",
    ),
) -> None:
    """Write working memory into a project file (AGENTS.md by default).

    Uses sentinel markers so repeated runs update only the memory section.
    Works across all AI IDEs: Claude Code reads CLAUDE.md, Codex reads AGENTS.md.
    """

    async def _inject():
        from .git import resolve_canonical_id
        from .memory_repo import (
            MEMORY_SECTIONS,
            list_scope_sections,
            render_section_text,
        )

        project_path = (project or Path(".")).resolve()
        content = ""

        canonical_id = ""
        try:
            canonical_id = resolve_canonical_id(project_path)
        except Exception:
            canonical_id = ""

        if canonical_id:
            sections = list_scope_sections(
                canonical_id=canonical_id,
                display_name=project_path.name,
                scope_path=str(project_path),
            )
            repo_parts = []
            for section in MEMORY_SECTIONS:
                entries = sections.get(section, [])
                rendered = render_section_text(entries)
                if rendered:
                    title = section.replace("_", " ").title()
                    repo_parts.append(f"# {title}\n\n{rendered}")
            if repo_parts:
                content = "\n\n".join(repo_parts)

        if not content:
            from .db.queries import get_working_memory
            from .db.schema import apply_schema

            db = _get_db()
            await db.connect()
            try:
                await apply_schema(db)
                scope = await _resolve_scope(db, str(project_path))
                if not scope:
                    console.print(f"[yellow]No scope found for:[/yellow] {project_path}")
                    return

                scope_id = str(scope["id"])
                memory = await get_working_memory(db, scope_id)
                if not memory:
                    console.print(
                        "[dim]No working memory yet. Run: uc memory refresh --project .[/dim]"
                    )
                    return
                content = memory.get("content", "")
            finally:
                await db.close()

        if not content:
            console.print(
                "[dim]No working memory content yet. Run: uc memory refresh --project .[/dim]"
            )
            return

        memory_block = f"{_MEMORY_START}\n{content}\n{_MEMORY_END}"

        target_path = project_path / target
        if target_path.exists():
            existing = target_path.read_text(encoding="utf-8")
            # Replace existing memory section or append
            if _MEMORY_START in existing:
                import re

                pattern = re.escape(_MEMORY_START) + r".*?" + re.escape(_MEMORY_END)
                new_content = re.sub(pattern, memory_block, existing, flags=re.DOTALL)
            else:
                new_content = existing.rstrip() + "\n\n" + memory_block + "\n"
        else:
            new_content = memory_block + "\n"

        target_path.write_text(new_content, encoding="utf-8")
        console.print(f"[green]Injected working memory into[/green] {target_path.name}")

    _run_async(_inject())


@memory_app.command("eject")
def memory_eject(
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (default: cwd)",
    ),
    target: str = typer.Option(
        "AGENTS.md",
        "--target",
        "-t",
        help="Target file to eject from",
    ),
) -> None:
    """Remove the working memory section from a project file."""
    import re

    project_path = (project or Path(".")).resolve()
    target_path = project_path / target

    if not target_path.exists():
        console.print(f"[dim]{target_path.name} does not exist.[/dim]")
        return

    existing = target_path.read_text(encoding="utf-8")
    if _MEMORY_START not in existing:
        console.print(f"[dim]No memory section found in {target_path.name}.[/dim]")
        return

    pattern = r"\n*" + re.escape(_MEMORY_START) + r".*?" + re.escape(_MEMORY_END) + r"\n*"
    cleaned = re.sub(pattern, "\n", existing, flags=re.DOTALL).strip()

    if cleaned:
        target_path.write_text(cleaned + "\n", encoding="utf-8")
    else:
        target_path.unlink()

    console.print(f"[green]Ejected working memory from[/green] {target_path.name}")


# --- Helpers ---


async def _resolve_scope(db: Any, project_path: str) -> dict[str, Any] | None:
    """Resolve a project path to a scope using multiple strategies.

    1. Git-aware canonical_id match
    2. Exact path match
    3. Name match (if input looks like a name)
    4. Parent directory walk
    """
    from .db.queries import find_scope_by_canonical_id, find_scope_by_name, find_scope_by_path
    from .git import resolve_canonical_id

    # 1. Git-aware canonical_id match
    try:
        canonical_id = resolve_canonical_id(Path(project_path))
        scope = await find_scope_by_canonical_id(db, canonical_id)
        if scope:
            return scope
    except Exception:
        pass  # Fall through to path-based resolution

    # 2. Exact path match
    scope = await find_scope_by_path(db, project_path)
    if scope:
        return scope

    # 2. Name match — try the basename
    basename = Path(project_path).name
    candidates = await find_scope_by_name(db, basename)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Pick the scope whose path is the longest prefix of project_path
        best = None
        best_len = -1
        for c in candidates:
            cp = c.get("path", "")
            if cp and project_path.startswith(cp) and len(cp) > best_len:
                best = c
                best_len = len(cp)
        if best:
            return best
        # Otherwise return the most recently created one
        return candidates[0]

    # 3. Parent directory walk
    for parent in Path(project_path).parents:
        parent_str = str(parent)
        scope = await find_scope_by_path(db, parent_str)
        if scope:
            return scope

    return None


async def _resolve_scope_ref(db: Any, ref: str) -> dict[str, Any] | None:
    """Resolve a scope reference (ID, name, or path) to a scope record."""
    from .db.queries import find_scope_by_name, find_scope_by_path, get_scope

    # Try as a scope ID first (e.g. "scope:abc123")
    if ref.startswith("scope:"):
        return await get_scope(db, ref)

    # Try as an exact path
    scope = await find_scope_by_path(db, ref)
    if scope:
        return scope

    # Try as a name
    candidates = await find_scope_by_name(db, ref)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Exact name match preferred
        for c in candidates:
            if c.get("name", "").lower() == ref.lower():
                return c
        return candidates[0]

    return None


def _check(label: str, ok: bool) -> None:
    icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {icon}  {label}")
