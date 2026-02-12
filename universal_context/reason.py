"""RLM (Recursive Language Model) integration for agentic memory exploration.

Provides the deep reasoning backend for `uc ask --deep` (and `uc admin reason`)
where the LLM programs its own exploration of UC's graph database via a
sandboxed REPL loop. Uses DSPy's RLM module with a custom in-process Python
interpreter and UC-specific tools.

Requires: ``pip install universal-context[reason]`` (installs DSPy)
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from io import StringIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import UCConfig
    from .db.client import UCDatabase
    from .embed import EmbedProvider

logger = logging.getLogger(__name__)


# ============================================================
# Async Bridge
# ============================================================


class AsyncBridge:
    """Bridge between sync tool calls and the main async event loop.

    When RLM runs in a worker thread (via ``asyncio.to_thread``), tools
    submit coroutines back to the main event loop with
    ``run_coroutine_threadsafe`` and block on the result.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def run(self, coro: Any, timeout: float = 30.0) -> Any:
        """Submit an async coroutine to the main loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        """No-op — we don't own the event loop."""


# ============================================================
# Local Interpreter
# ============================================================


class _SubmitError(Exception):
    """Raised when ``SUBMIT()`` is called inside the REPL."""

    def __init__(self, value: Any) -> None:
        self.value = value


class _FinalOutput:
    """Fallback ``FinalOutput`` when DSPy doesn't export one."""

    def __init__(self, output: Any) -> None:
        self.output = output

    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"


def _get_final_output_class() -> type:
    """Import DSPy's ``FinalOutput`` or fall back to our own."""
    for path in (
        "dspy.tools.python_interpreter",
        "dspy.predict.rlm",
        "dspy.predict.code_act",
    ):
        try:
            mod = __import__(path, fromlist=["FinalOutput"])
            cls = getattr(mod, "FinalOutput", None)
            if cls is not None:
                return cls
        except (ImportError, AttributeError):
            continue
    return _FinalOutput


class LocalInterpreter:
    """In-process Python interpreter for DSPy RLM.

    Executes code in a persistent namespace with captured stdout,
    restricted imports, and tool injection. No Deno/WASM dependency.

    - **Persistent namespace**: variables survive across ``execute()`` calls
    - **Tools**: injected as bare functions callable by name
    - **print()**: captured via ``StringIO``
    - **SUBMIT(value)**: raises internal signal → returns ``FinalOutput``
    - **Restricted imports**: blocks ``os``, ``subprocess``, ``sys``, etc.
    """

    BLOCKED_MODULES = frozenset(
        {
            "os",
            "subprocess",
            "sys",
            "shutil",
            "socket",
            "pathlib",
            "signal",
            "ctypes",
            "multiprocessing",
        }
    )

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._timeout = timeout
        self._namespace: dict[str, Any] = {}
        self._tools = tools or {}
        self._namespace.update(self._tools)
        self._namespace["SUBMIT"] = self._submit
        self._namespace["__builtins__"] = self._make_restricted_builtins()

    @property
    def tools(self) -> dict[str, Any]:
        """Tools available for interpreter code to call (CodeInterpreter protocol)."""
        return self._tools

    def _submit(self, value: Any) -> None:
        raise _SubmitError(value)

    def _make_restricted_builtins(self) -> dict[str, Any]:
        import builtins

        safe = dict(vars(builtins))
        orig_import = builtins.__import__
        blocked = self.BLOCKED_MODULES

        def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            top = name.split(".")[0]
            if top in blocked:
                raise ImportError(
                    f"Import of '{name}' is blocked for security. Use the provided tools instead."
                )
            return orig_import(name, *args, **kwargs)

        safe["__import__"] = restricted_import
        return safe

    def start(self) -> None:
        """No-op — in-process interpreter needs no setup."""

    def shutdown(self) -> None:
        """Clear the namespace."""
        self._namespace.clear()

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute Python code in the persistent namespace.

        Returns captured stdout output, or ``FinalOutput`` if SUBMIT was called.
        On errors, returns a formatted error string (doesn't raise).
        """
        if variables:
            self._namespace.update(variables)

        stdout_capture = StringIO()
        self._namespace["print"] = lambda *a, **kw: print(*a, file=stdout_capture, **kw)

        try:
            exec(compile(code, "<rlm>", "exec"), self._namespace)
            output = stdout_capture.getvalue()
            return output.rstrip() if output else ""
        except _SubmitError as sig:
            final_cls = _get_final_output_class()
            return final_cls(sig.value)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)


# ============================================================
# LM Bridge
# ============================================================


def resolve_reason_llm_target(config: UCConfig) -> tuple[str, str, str]:
    """Resolve provider/model/key for DSPy RLM from UC config."""
    provider = (config.llm_provider or "auto").strip().lower()
    raw_model = (config.llm_model or "").strip()

    if provider == "auto":
        for candidate in ("openrouter", "claude", "openai"):
            if config.get_api_key(candidate):
                provider = candidate
                break
        else:
            raise ValueError("No LLM API key configured for any provider")

    api_key = config.get_api_key(provider)
    if not api_key:
        raise ValueError(f"No API key for provider '{provider}'")

    defaults = {
        "openrouter": "anthropic/claude-haiku-4-5-20251001",
        "claude": "claude-haiku-4-5-20251001",
        "openai": "gpt-4.1-mini",
    }

    if provider == "openrouter":
        model = raw_model or defaults["openrouter"]
        if model.startswith("openrouter/"):
            model = model.removeprefix("openrouter/")
        return provider, f"openrouter/{model}", api_key

    if provider == "claude":
        model = raw_model or defaults["claude"]
        if model.startswith("anthropic/"):
            model = model.removeprefix("anthropic/")
        return provider, f"anthropic/{model}", api_key

    if provider == "openai":
        model = raw_model or defaults["openai"]
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")
        return provider, f"openai/{model}", api_key

    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_dspy_lm(
    config: UCConfig,
    resolved: tuple[str, str, str] | None = None,
) -> Any:
    """Build a ``dspy.LM`` instance from UC config."""
    import dspy

    provider, lm_model, api_key = resolved or resolve_reason_llm_target(config)

    if provider == "openrouter":
        return dspy.LM(lm_model, api_key=api_key, api_base="https://openrouter.ai/api/v1")

    if provider in {"claude", "openai"}:
        return dspy.LM(lm_model, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ============================================================
# Tools
# ============================================================


_WRITE_PATTERN = re.compile(
    r"\b(CREATE|UPDATE|DELETE|RELATE|DEFINE|REMOVE|INSERT|UPSERT)\b",
    re.IGNORECASE,
)
_RECORD_ID_PATTERN = re.compile(r"\b(?:scope|run|turn|artifact):[a-z0-9]{3,}\b", re.IGNORECASE)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif item is not None:
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _extract_record_ids(text: str) -> list[str]:
    if not text:
        return []
    return _dedupe_keep_order([m.group(0) for m in _RECORD_ID_PATTERN.finditer(text)])


def _parse_obj_literal(raw: str) -> Any:
    """Parse a JSON/Python object literal. Returns raw string on failure."""
    raw = raw.strip()
    if not raw:
        return raw

    # JSON first (strict), then Python literal fallback
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _extract_submit_arg(text: str) -> str | None:
    """Extract the first SUBMIT(...) argument from a text blob, if present."""
    marker = "SUBMIT("
    start = text.find(marker)
    if start < 0:
        return None

    i = start + len(marker)
    depth = 1
    arg_start = i
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[arg_start:i].strip()
        i += 1
    return None


def _normalize_reasoning_output(raw_output: Any) -> dict[str, Any]:
    """Normalize RLM output into a typed contract.

    Output contract:
      - answer: str
      - facts: list[str]
      - decisions: list[str]
      - open_questions: list[str]
      - evidence_ids: list[str]
    """
    parsed: Any = raw_output

    if isinstance(raw_output, str):
        submit_arg = _extract_submit_arg(raw_output)
        if submit_arg:
            parsed = _parse_obj_literal(submit_arg)
        elif raw_output.strip().startswith("{"):
            parsed = _parse_obj_literal(raw_output)

    answer = ""
    facts: list[str] = []
    decisions: list[str] = []
    open_questions: list[str] = []
    evidence_ids: list[str] = []

    if isinstance(parsed, dict):
        answer = str(
            parsed.get("answer")
            or parsed.get("summary")
            or parsed.get("response")
            or ""
        ).strip()
        facts = _as_string_list(parsed.get("facts"))
        decisions = _as_string_list(parsed.get("decisions"))
        open_questions = _as_string_list(
            parsed.get("open_questions") or parsed.get("questions"),
        )
        evidence_ids = _as_string_list(
            parsed.get("evidence_ids")
            or parsed.get("evidence")
            or parsed.get("sources"),
        )
    else:
        answer = str(parsed).strip()

    text_sources = [answer, *facts, *decisions, *open_questions]
    for src in text_sources:
        evidence_ids.extend(_extract_record_ids(src))

    evidence_ids = _dedupe_keep_order(evidence_ids)

    return {
        "answer": answer,
        "facts": _dedupe_keep_order(facts),
        "decisions": _dedupe_keep_order(decisions),
        "open_questions": _dedupe_keep_order(open_questions),
        "evidence_ids": evidence_ids,
    }


def build_tools(
    db: UCDatabase,
    bridge: AsyncBridge,
    scope_id: str | None,
    embed_provider: EmbedProvider | None,
) -> dict[str, Any]:
    """Build the 6 RLM tool functions that close over DB and bridge."""
    from .db.queries import (
        get_turn_summaries,
        list_runs,
        search_artifacts,
        semantic_search,
    )
    from .db.queries import (
        get_working_memory as _get_working_memory,
    )

    def get_working_memory() -> str:  # noqa: F811
        """Get the project's distilled working memory.

        This is an LLM-distilled summary of the project's entire history.
        START HERE — it gives you a high-level overview before diving deeper.
        Returns the working memory content as a string.
        """
        if not scope_id:
            return "No project scope resolved."
        memory = bridge.run(_get_working_memory(db, scope_id))
        if not memory:
            return "No working memory available for this project."
        return memory.get("content", "(empty)")

    def search_sessions(query: str, limit: int = 10) -> str:
        """Search session summaries using BM25 keyword search.

        Args:
            query: Search keywords (e.g. "authentication", "database migration")
            limit: Max results (default 10)

        Returns matched summaries with their IDs and content.
        """
        results = bridge.run(
            search_artifacts(db, query, kind="summary", limit=limit, scope_id=scope_id)
        )
        if not results:
            return f"No results for query: {query}"
        lines = []
        for r in results:
            rid = str(r.get("id", ""))
            content = (r.get("content") or "")[:300]
            lines.append(f"[{rid}] {content}")
        return "\n---\n".join(lines)

    def search_semantic(context: str, limit: int = 10) -> str:
        """Search session summaries using embedding-based semantic similarity.

        Args:
            context: Describe what you're looking for in natural language
            limit: Max results (default 10)

        Returns semantically similar summaries ranked by relevance.
        """
        if embed_provider is None:
            return "Semantic search unavailable — no embedding provider configured."
        query_vec = bridge.run(embed_provider.embed_query(context))
        results = bridge.run(
            semantic_search(db, query_vec, kind="summary", limit=limit, scope_id=scope_id)
        )
        if not results:
            return f"No semantic matches for: {context}"
        lines = []
        for r in results:
            rid = str(r.get("id", ""))
            score = r.get("score", r.get("dist", 0))
            content = (r.get("content") or "")[:300]
            lines.append(f"[{rid}] (score: {score:.3f}) {content}")
        return "\n---\n".join(lines)

    def list_recent_runs(limit: int = 10, branch: str = "") -> str:
        """List recent runs (coding sessions) for this project.

        Args:
            limit: Max runs to return (default 10)
            branch: Filter by git branch name (optional, empty = all branches)

        Returns run IDs, agent types, branches, status, and timestamps.
        """
        runs = bridge.run(list_runs(db, scope_id=scope_id, branch=branch or None, limit=limit))
        if not runs:
            return "No runs found."
        lines = []
        for r in runs:
            rid = str(r.get("id", ""))
            agent = r.get("agent_type", "?")
            b = r.get("branch", "")
            status = r.get("status", "?")
            started = str(r.get("started_at", ""))[:19]
            merged = r.get("merged_to", "")
            header = f"[{rid}] {agent} on {b}" if b else f"[{rid}] {agent}"
            parts = [header, f"  status={status} started={started}"]
            if merged:
                parts.append(f"  merged_to={merged}")
            lines.append("\n".join(parts))
        return "\n---\n".join(lines)

    def get_run_turns(run_id: str, limit: int = 10) -> str:
        """Get turn-by-turn details for a specific run.

        Args:
            run_id: Run ID (e.g. "run:abc123def456")
            limit: Max turns to return (default 10)

        Returns each turn's sequence number, user message, and summary.
        """
        summaries = bridge.run(get_turn_summaries(db, run_id, limit=limit))
        if not summaries:
            return f"No turns found for run: {run_id}"
        lines = []
        for s in summaries:
            seq = s.get("sequence", "?")
            msg = (s.get("user_message") or "(no message)")[:200]
            summary = (s.get("summary") or "(no summary)")[:300]
            lines.append(f"Turn #{seq}\n  User: {msg}\n  Summary: {summary}")
        return "\n---\n".join(lines)

    def query_graph(surql: str) -> str:
        """Execute a read-only SurrealQL query against the graph database.

        ONLY SELECT queries are allowed. CREATE/UPDATE/DELETE/RELATE/DEFINE/REMOVE
        are rejected for safety.

        Schema reference:
          Tables: scope, run, turn, artifact, job
          Edges: contains (scope->run, run->turn), produced (turn->artifact),
                 depends_on (artifact->artifact)
          Key fields:
            scope: name, path, canonical_id
            run: scope, agent_type, status, branch, commit_sha, merged_to, started_at
            turn: run, sequence, user_message, started_at
            artifact: kind, content, scope, embedding, metadata
          artifact.kind values: "transcript", "summary", "working_memory"

        Args:
            surql: A SurrealQL SELECT query

        Returns query results as formatted text (max 5000 chars).
        """
        if _WRITE_PATTERN.search(surql):
            return "ERROR: Write operations are not allowed. Use SELECT queries only."
        try:
            results = bridge.run(db.query(surql))
        except Exception as e:
            return f"Query error: {e}"
        if not results:
            return "(empty result set)"
        output = _format_query_results(results)
        if len(output) > 5000:
            output = output[:5000] + "\n... (truncated)"
        return output

    return {
        "get_working_memory": get_working_memory,
        "search_sessions": search_sessions,
        "search_semantic": search_semantic,
        "list_recent_runs": list_recent_runs,
        "get_run_turns": get_run_turns,
        "query_graph": query_graph,
    }


def _format_query_results(results: list[Any]) -> str:
    """Format SurrealDB query results for LLM consumption."""
    import json

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "__class__") and "RecordID" in type(obj).__name__:
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        return obj

    serialized = _serialize(results)
    try:
        return json.dumps(serialized, indent=2, default=str)
    except (TypeError, ValueError):
        return str(serialized)


# ============================================================
# Orchestrator
# ============================================================


_RLM_INSTRUCTIONS = """\
You are exploring a project's AI coding session history stored in a graph database.

AVAILABLE TOOLS (call these as regular Python functions):
- get_working_memory() → distilled project overview
- search_sessions(query, limit=10) → BM25 keyword search across summaries
- search_semantic(context, limit=10) → embedding-based similarity search
- list_recent_runs(limit=10, branch="") → list recent coding sessions
- get_run_turns(run_id, limit=10) → turn-by-turn details for a specific run
- query_graph(surql) → read-only SurrealQL SELECT queries

STRATEGY:
1. Start by calling get_working_memory() for a high-level project overview
2. Use search_sessions() for keyword-based search across session summaries
3. Use search_semantic() for meaning-based search when keywords aren't precise enough
4. Use list_recent_runs() to see recent coding sessions
5. Use get_run_turns() to drill into specific sessions for details
6. Use query_graph() for advanced SurrealQL queries to traverse the graph

IMPORTANT:
- Only call the 6 tools listed above. No other functions are available.
- Always start with get_working_memory() for project context
- Search broadly first, then drill into specific runs/turns
- query_graph() only accepts SELECT statements (no writes)
- When you have enough information, call:
  SUBMIT({
    "answer": "...markdown...",
    "facts": ["..."],
    "decisions": ["..."],
    "open_questions": ["..."],
    "evidence_ids": ["artifact:...", "run:..."]
  })
- `evidence_ids` should contain concrete UC IDs you used as evidence
- Keep `facts/decisions/open_questions` concise and atomic
- Use markdown formatting in `answer`
- Be specific — cite run IDs, turn numbers, and file names when possible
"""


async def reason(
    question: str,
    project_path: str,
    config: UCConfig,
    max_iterations: int = 12,
    max_llm_calls: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run agentic RLM reasoning over UC's graph database.

    The LLM explores the project's history through a REPL loop,
    using tools to search, navigate, and query the graph database.

    Returns dict with keys:
      answer, facts, decisions, open_questions, evidence_ids,
      trajectory, iterations, scope, llm_provider, llm_model
    """
    import dspy

    resolved_llm = resolve_reason_llm_target(config)
    lm_provider, lm_model, _ = resolved_llm
    lm = build_dspy_lm(config, resolved=resolved_llm)
    dspy.configure(lm=lm)

    from pathlib import Path

    from .cli import _resolve_scope
    from .db.client import UCDatabase
    from .db.schema import apply_schema
    from .embed import create_embed_provider

    if config.db_url:
        db = UCDatabase.from_url(config.db_url, config.db_user, config.db_pass)
    else:
        db = UCDatabase.from_path(Path(config.resolved_db_path))

    loop = asyncio.get_running_loop()
    bridge = AsyncBridge(loop)
    interpreter: LocalInterpreter | None = None

    try:
        await db.connect()
        await apply_schema(db)

        scope = await _resolve_scope(db, project_path)
        scope_id = str(scope["id"]) if scope else None

        embed_provider = await create_embed_provider(config)

        tools = build_tools(db, bridge, scope_id, embed_provider)

        interpreter = LocalInterpreter(tools=tools)

        scope_name = scope.get("name", "") if scope else "unknown"
        project_context = f"Project: {scope_name} (scope: {scope_id or 'none'})"

        sig = dspy.Signature(
            "question, project_context -> answer",
            instructions=_RLM_INSTRUCTIONS,
        )

        rlm = dspy.RLM(
            signature=sig,
            tools=list(tools.values()),
            interpreter=interpreter,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=verbose,
        )

        # RLM forward() is sync — run in thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            rlm,
            question=question,
            project_context=project_context,
        )

        trajectory = getattr(result, "trajectory", [])
        structured = _normalize_reasoning_output(result.answer)

        # Evidence IDs may also appear in tool outputs captured in the trajectory.
        if trajectory:
            trajectory_text = str(trajectory)
            merged_ids = structured["evidence_ids"] + _extract_record_ids(trajectory_text)
            structured["evidence_ids"] = _dedupe_keep_order(merged_ids)

        return {
            **structured,
            "trajectory": trajectory,
            "iterations": getattr(result, "iterations", None),
            "scope": scope_id,
            "llm_provider": lm_provider,
            "llm_model": lm_model,
        }
    finally:
        bridge.shutdown()
        if interpreter is not None:
            interpreter.shutdown()
        await db.close()
