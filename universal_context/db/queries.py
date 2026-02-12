"""Reusable SurrealQL query helpers for Universal Context.

All multi-record write operations use SurrealQL BEGIN/COMMIT transactions
for atomicity. The Python SDK's client-side transactions are not available
in embedded mode, but SurrealQL transactions work fine.

Note: RecordID objects from the SDK must be converted to strings with str()
for use in subsequent queries.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from .client import UCDatabase


def _gen_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


def _id_str(record: dict[str, Any]) -> str:
    """Extract the string form of a record's ID (e.g. 'scope:abc123')."""
    return str(record["id"])


# ============================================================
# SCOPE
# ============================================================


async def create_scope(
    db: UCDatabase,
    name: str,
    path: str | None = None,
    canonical_id: str | None = None,
) -> dict[str, Any]:
    """Create a new scope."""
    sid = _gen_id()
    canonical_clause = ", canonical_id = $canonical_id" if canonical_id else ""
    result = await db.query(
        f"CREATE scope:{sid} SET name = $name, path = $path{canonical_clause}",
        {"name": name, "path": path, "canonical_id": canonical_id},
    )
    return result[0] if result else {}


async def get_scope(db: UCDatabase, scope_id: str) -> dict[str, Any] | None:
    """Get a scope by ID."""
    result = await db.query(f"SELECT * FROM {scope_id}")
    return result[0] if result else None


async def list_scopes(db: UCDatabase) -> list[dict[str, Any]]:
    """List all scopes."""
    return await db.query("SELECT * FROM scope ORDER BY created_at DESC")


async def find_scope_by_path(db: UCDatabase, path: str) -> dict[str, Any] | None:
    """Find a scope by its filesystem path."""
    result = await db.query("SELECT * FROM scope WHERE path = $path LIMIT 1", {"path": path})
    return result[0] if result else None


async def find_scope_by_canonical_id(
    db: UCDatabase,
    canonical_id: str,
) -> dict[str, Any] | None:
    """Find a scope by its canonical identity (git remote, git-local://, or path://)."""
    result = await db.query(
        "SELECT * FROM scope WHERE canonical_id = $cid LIMIT 1",
        {"cid": canonical_id},
    )
    return result[0] if result else None


async def find_scope_by_name(
    db: UCDatabase,
    name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find scopes by name substring (case-insensitive)."""
    return await db.query(
        "SELECT * FROM scope WHERE string::lowercase(name) CONTAINS string::lowercase($name) "
        "ORDER BY created_at DESC LIMIT $limit",
        {"name": name, "limit": limit},
    )


async def update_scope(
    db: UCDatabase,
    scope_id: str,
    name: str | None = None,
    path: str | None = None,
    canonical_id: str | None = None,
) -> dict[str, Any] | None:
    """Update a scope's name, path, and/or canonical_id."""
    sets: list[str] = []
    params: dict[str, Any] = {}
    if name is not None:
        sets.append("name = $name")
        params["name"] = name
    if path is not None:
        sets.append("path = $path")
        params["path"] = path
    if canonical_id is not None:
        sets.append("canonical_id = $canonical_id")
        params["canonical_id"] = canonical_id
    if not sets:
        return await get_scope(db, scope_id)
    result = await db.query(f"UPDATE {scope_id} SET {', '.join(sets)}", params)
    return result[0] if result else None


async def delete_scope(db: UCDatabase, scope_id: str) -> None:
    """Delete a scope and cascade to its runs, turns, artifacts, and jobs.

    Walks the graph: scope -> runs -> turns -> artifacts, plus edges and jobs.
    """
    run_rows = await db.query(f"SELECT id FROM run WHERE scope = {scope_id}")
    run_ids = [str(r["id"]) for r in run_rows if isinstance(r, dict) and r.get("id")]

    turn_ids: list[str] = []
    for rid in run_ids:
        turns = await db.query(f"SELECT id FROM turn WHERE run = {rid}")
        turn_ids.extend(
            str(t["id"]) for t in turns if isinstance(t, dict) and t.get("id")
        )

    # Include all artifacts denormalized to this scope (transcript, summary, working_memory, etc.)
    artifact_rows = await db.query(f"SELECT id FROM artifact WHERE scope = {scope_id}")
    artifact_ids: set[str] = {
        str(a["id"]) for a in artifact_rows if isinstance(a, dict) and a.get("id")
    }

    # Also include artifacts produced by scope turns in case legacy data lacks artifact.scope
    for tid in turn_ids:
        produced = await db.query(f"SELECT out FROM produced WHERE in = {tid}")
        for edge in produced if isinstance(produced, list) else []:
            out = edge.get("out") if isinstance(edge, dict) else None
            if out:
                artifact_ids.add(str(out))

    # Remove jobs targeting this scope or any descendant records
    await db.query("DELETE FROM job WHERE target = $target", {"target": scope_id})
    for rid in run_ids:
        await db.query("DELETE FROM job WHERE target = $target", {"target": rid})
    for tid in turn_ids:
        await db.query("DELETE FROM job WHERE target = $target", {"target": tid})

    # Delete checkpoint records tied to this scope's runs/turns
    checkpoint_ids: set[str] = set()
    for rid in run_ids:
        cps = await db.query(f"SELECT id FROM checkpoint WHERE run = {rid}")
        checkpoint_ids.update(
            str(c["id"]) for c in cps if isinstance(c, dict) and c.get("id")
        )
    for tid in turn_ids:
        cps = await db.query(f"SELECT id FROM checkpoint WHERE turn = {tid}")
        checkpoint_ids.update(
            str(c["id"]) for c in cps if isinstance(c, dict) and c.get("id")
        )

    # Remove graph edges before record deletes
    for aid in artifact_ids:
        await db.query(f"DELETE FROM depends_on WHERE in = {aid} OR out = {aid}")
        await db.query(f"DELETE FROM produced WHERE out = {aid}")
    for tid in turn_ids:
        await db.query(f"DELETE FROM produced WHERE in = {tid}")
        await db.query(f"DELETE FROM contains WHERE out = {tid}")
        await db.query(f"DELETE FROM checkpoint_at WHERE in = {tid} OR out = {tid}")
    for rid in run_ids:
        await db.query(f"DELETE FROM contains WHERE out = {rid}")
        await db.query(f"DELETE FROM checkpoint_at WHERE in = {rid} OR out = {rid}")

    # Delete records from leaves upward
    for cid in checkpoint_ids:
        await db.query(f"DELETE {cid}")
    for aid in artifact_ids:
        await db.query(f"DELETE {aid}")
    for tid in turn_ids:
        await db.query(f"DELETE {tid}")
    for rid in run_ids:
        await db.query(f"DELETE {rid}")

    await db.query(f"DELETE FROM contains WHERE in = {scope_id}")
    await db.query(f"DELETE {scope_id}")


async def merge_scopes(
    db: UCDatabase,
    source_id: str,
    target_id: str,
) -> None:
    """Move all runs and artifacts from source scope into target, then delete source."""
    moved_runs = await db.query(f"SELECT id FROM run WHERE scope = {source_id}")
    moved_run_ids = [str(r["id"]) for r in moved_runs if isinstance(r, dict) and r.get("id")]

    # Re-point runs
    await db.query(f"UPDATE run SET scope = {target_id} WHERE scope = {source_id}")
    # Re-point artifacts
    await db.query(f"UPDATE artifact SET scope = {target_id} WHERE scope = {source_id}")
    # Re-point working memory metadata references
    await db.query(
        "UPDATE artifact SET metadata.scope_id = $tid "
        "WHERE kind = 'working_memory' AND metadata.scope_id = $sid",
        {"tid": str(target_id), "sid": str(source_id)},
    )
    # Move graph edges: delete old source edges, then add target edges for moved runs only
    await db.query(f"DELETE FROM contains WHERE in = {source_id}")
    for rid in moved_run_ids:
        existing = await db.query(
            f"SELECT id FROM contains WHERE in = {target_id} AND out = {rid} LIMIT 1"
        )
        if not existing:
            await db.query(f"RELATE {target_id}->contains->{rid}")
    # Delete the source scope
    await db.query(f"DELETE {source_id}")


async def list_scopes_with_stats(db: UCDatabase) -> list[dict[str, Any]]:
    """List scopes with run count, turn count, and last activity."""
    scopes = await db.query("SELECT * FROM scope ORDER BY created_at DESC")
    results = []
    for s in scopes:
        sid = str(s["id"])
        # Count runs
        run_count_result = await db.query(f"SELECT count() FROM run WHERE scope = {sid} GROUP ALL")
        run_count = run_count_result[0].get("count", 0) if run_count_result else 0

        # Count turns across all runs
        turn_count_result = await db.query(
            f"SELECT count() FROM turn WHERE run.scope = {sid} GROUP ALL"
        )
        turn_count = turn_count_result[0].get("count", 0) if turn_count_result else 0

        # Last activity (most recent run start)
        last_run = await db.query(
            f"SELECT started_at FROM run WHERE scope = {sid} ORDER BY started_at DESC LIMIT 1"
        )
        last_activity = last_run[0].get("started_at") if last_run else None

        # Agent type breakdown
        agents = await db.query(
            f"SELECT agent_type, count() FROM run WHERE scope = {sid} GROUP BY agent_type"
        )
        agent_breakdown = {a["agent_type"]: a["count"] for a in agents} if agents else {}

        results.append(
            {
                **s,
                "run_count": run_count,
                "turn_count": turn_count,
                "last_activity": last_activity,
                "agent_breakdown": agent_breakdown,
            }
        )
    return results


# ============================================================
# RUN
# ============================================================


async def create_run(
    db: UCDatabase,
    scope_id: str,
    agent_type: str,
    session_path: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Create a new run and link it to its scope via RELATE."""
    rid = _gen_id()
    extras = ""
    if branch:
        extras += ", branch = $branch"
    if commit_sha:
        extras += ", commit_sha = $commit_sha"
    result = await db.query(
        f"CREATE run:{rid} SET scope = {scope_id}, agent_type = $agent_type, "
        f'status = "active", session_path = $session_path{extras}',
        {
            "agent_type": agent_type,
            "session_path": session_path,
            "branch": branch,
            "commit_sha": commit_sha,
        },
    )
    run = result[0] if result else {}
    if run:
        await db.query(f"RELATE {scope_id}->contains->run:{rid}")
    return run


async def end_run(db: UCDatabase, run_id: str, status: str = "completed") -> None:
    """Mark a run as ended."""
    await db.query(
        f"UPDATE {run_id} SET ended_at = time::now(), status = $status",
        {"status": status},
    )


async def get_run(db: UCDatabase, run_id: str) -> dict[str, Any] | None:
    """Get a run by ID."""
    result = await db.query(f"SELECT * FROM {run_id}")
    return result[0] if result else None


async def list_runs(
    db: UCDatabase,
    scope_id: str | None = None,
    status: str | None = None,
    branch: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List runs, optionally filtered by scope, status, and/or branch."""
    conditions = []
    params: dict[str, Any] = {}
    if scope_id:
        conditions.append(f"scope = {scope_id}")
    if status:
        conditions.append("status = $status")
        params["status"] = status
    if branch:
        conditions.append("branch = $branch")
        params["branch"] = branch

    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    return await db.query(
        f"SELECT * FROM run{where} ORDER BY started_at DESC LIMIT {limit}", params
    )


async def find_runs_by_session_path(
    db: UCDatabase, session_path: str,
) -> list[dict[str, Any]]:
    """Find all runs for a session_path, newest first."""
    result = await db.query(
        "SELECT * FROM run WHERE session_path = $path ORDER BY started_at DESC",
        {"path": session_path},
    )
    return result if isinstance(result, list) else []


# ============================================================
# TURN (with atomic provenance)
# ============================================================


async def create_turn_with_artifact(
    db: UCDatabase,
    run_id: str,
    sequence: int,
    user_message: str | None,
    raw_content: str,
    create_summary_job: bool = True,
) -> dict[str, Any]:
    """Atomically create a turn + raw transcript artifact + provenance edges + optional job.

    Uses SurrealQL transaction for atomicity. Resolves scope from the
    run record and sets it on the transcript artifact.
    Returns dict with turn_id and artifact_id.
    """
    tid = _gen_id()
    aid = _gen_id()
    content_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:16]

    # Resolve scope from run (outside transaction — read-only)
    run = await get_run(db, run_id)
    scope_clause = ""
    if run:
        run_scope = run.get("scope")
        if run_scope:
            scope_clause = f", scope = {run_scope}"

    # Build the transaction
    stmts = [
        f"CREATE turn:{tid} SET run = {run_id}, sequence = $sequence, user_message = $user_message",
        f'CREATE artifact:{aid} SET kind = "transcript", content = $raw_content, '
        f"content_hash = $content_hash{scope_clause}",
        f"RELATE {run_id}->contains->turn:{tid} SET sequence = $sequence",
        f"RELATE turn:{tid}->produced->artifact:{aid}",
    ]

    if create_summary_job:
        jid = _gen_id()
        stmts.append(
            f'CREATE job:{jid} SET job_type = "turn_summary", status = "pending", '
            f'target = "turn:{tid}", priority = 0'
        )

    txn = "BEGIN TRANSACTION;\n" + ";\n".join(stmts) + ";\nCOMMIT TRANSACTION"
    await db.query(
        txn,
        {
            "sequence": sequence,
            "user_message": user_message,
            "raw_content": raw_content,
            "content_hash": content_hash,
        },
    )

    return {"turn_id": f"turn:{tid}", "artifact_id": f"artifact:{aid}"}


async def get_turn(db: UCDatabase, turn_id: str) -> dict[str, Any] | None:
    """Get a turn by ID."""
    result = await db.query(f"SELECT * FROM {turn_id}")
    return result[0] if result else None


async def list_turns(db: UCDatabase, run_id: str) -> list[dict[str, Any]]:
    """List all turns in a run, ordered by sequence."""
    return await db.query(f"SELECT * FROM turn WHERE run = {run_id} ORDER BY sequence ASC")


async def count_turns(db: UCDatabase, run_id: str) -> int:
    """Count turns in a run."""
    result = await db.query(f"SELECT count() FROM turn WHERE run = {run_id} GROUP ALL")
    return result[0].get("count", 0) if result else 0


# ============================================================
# ARTIFACT
# ============================================================


async def create_artifact(
    db: UCDatabase,
    kind: str,
    content: str | None = None,
    blob_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    scope_id: str | None = None,
) -> str:
    """Create a standalone artifact. Returns the artifact ID string."""
    aid = _gen_id()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16] if content else None
    scope_clause = f", scope = {scope_id}" if scope_id else ""
    await db.query(
        f"CREATE artifact:{aid} SET kind = $kind, content = $content, "
        f"content_hash = $content_hash, blob_path = $blob_path, metadata = $metadata"
        f"{scope_clause}",
        {
            "kind": kind,
            "content": content,
            "content_hash": content_hash,
            "blob_path": blob_path,
            "metadata": metadata or {},
        },
    )
    return f"artifact:{aid}"


async def create_derived_artifact(
    db: UCDatabase,
    kind: str,
    content: str,
    source_id: str,
    relationship: str = "derived_from",
    scope_id: str | None = None,
) -> str:
    """Create a derived artifact with a depends_on edge to its source."""
    aid = _gen_id()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    scope_clause = f", scope = {scope_id}" if scope_id else ""
    # Create artifact then edge
    await db.query(
        f"CREATE artifact:{aid} SET kind = $kind, content = $content, "
        f"content_hash = $content_hash{scope_clause}",
        {"kind": kind, "content": content, "content_hash": content_hash},
    )
    await db.query(
        f"RELATE artifact:{aid}->depends_on->{source_id} SET relationship = $rel",
        {"rel": relationship},
    )
    return f"artifact:{aid}"


async def search_artifacts(
    db: UCDatabase,
    query: str,
    kind: str | None = None,
    limit: int = 20,
    scope_id: str | None = None,
) -> list[dict[str, Any]]:
    """Full-text search across artifacts.

    On v3 server: uses BM25 FTS index (FULLTEXT ANALYZER).
    On embedded: falls back to brute-force substring match.
    """
    params: dict[str, Any] = {"query": query}
    conditions: list[str] = []

    if kind:
        conditions.append("kind = $kind")
        params["kind"] = kind
    if scope_id:
        conditions.append(f"scope = {scope_id}")

    if db.is_server:
        conditions.insert(0, "content @@ $query")
    else:
        # Embedded fallback: case-insensitive substring match
        conditions.insert(0, "string::lowercase(content) CONTAINS string::lowercase($query)")
        conditions.insert(0, "content != NONE")

    where = " AND ".join(conditions)
    result = await db.query(f"SELECT * FROM artifact WHERE {where} LIMIT {limit}", params)
    return result if isinstance(result, list) else []


# ============================================================
# PROVENANCE
# ============================================================


async def store_embedding(db: UCDatabase, artifact_id: str, embedding: list[float]) -> None:
    """Store an embedding vector on an artifact record."""
    await db.query(
        f"UPDATE {artifact_id} SET embedding = $embedding",
        {"embedding": embedding},
    )


async def semantic_search(
    db: UCDatabase,
    query_embedding: list[float],
    kind: str | None = None,
    limit: int = 10,
    scope_id: str | None = None,
) -> list[dict[str, Any]]:
    """Find artifacts by vector similarity.

    On v3 server: tries HNSW KNN first (fast). If the index is stale
    (SurrealDB v3 beta build-once bug) and returns 0 results, falls
    back to brute-force cosine — same path used by embedded mode.
    """
    params: dict[str, Any] = {"qvec": query_embedding}
    extra_filters: list[str] = []
    if kind:
        extra_filters.append("kind = $kind")
        params["kind"] = kind
    if scope_id:
        extra_filters.append(f"scope = {scope_id}")

    # Try HNSW KNN on server (fast but may be stale after inserts)
    if db.is_server:
        extra = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""
        result = await db.query(
            "SELECT *, vector::distance::knn() AS dist"
            f" FROM artifact WHERE embedding <|{limit},40|> $qvec{extra}"
            " ORDER BY dist",
            params,
        )
        if isinstance(result, list) and result:
            return result
        # HNSW stale or empty — fall through to brute-force cosine

    # Brute-force cosine similarity (embedded, or HNSW fallback on server)
    extra_filters.insert(0, "embedding != NONE")
    where = " AND ".join(extra_filters)
    result = await db.query(
        "SELECT *, vector::similarity::cosine(embedding, $qvec) AS score"
        f" FROM artifact WHERE {where}"
        f" ORDER BY score DESC LIMIT {limit}",
        params,
    )
    return result if isinstance(result, list) else []


async def hybrid_search(
    db: UCDatabase,
    query_text: str,
    query_embedding: list[float],
    kind: str | None = None,
    limit: int = 10,
    scope_id: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search: fuse full-text + vector results via RRF.

    On v3 server: uses SurrealDB's search::rrf() with BM25 + HNSW.
    On embedded: runs text + vector fallbacks and merges via Python RRF.
    """
    if db.is_server:
        extra_filters: list[str] = []
        if kind:
            extra_filters.append("kind = $kind")
        if scope_id:
            extra_filters.append(f"scope = {scope_id}")
        extra = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""

        params: dict[str, Any] = {"query": query_text, "qvec": query_embedding}
        if kind:
            params["kind"] = kind

        result = await db.query(
            f"LET $ft = SELECT id FROM artifact WHERE content @@ $query{extra};"
            f" LET $vs = SELECT id FROM artifact WHERE embedding <|{limit},40|> $qvec"
            f"{extra};"
            f" search::rrf([$ft, $vs], {limit}, 60)",
            params,
        )
        return result if isinstance(result, list) else []

    # Embedded fallback: run both searches, merge via Python RRF
    text_results = await search_artifacts(
        db,
        query_text,
        kind=kind,
        limit=limit,
        scope_id=scope_id,
    )
    vector_results = await semantic_search(
        db,
        query_embedding,
        kind=kind,
        limit=limit,
        scope_id=scope_id,
    )
    return _merge_rrf(text_results, vector_results, limit=limit)


def _merge_rrf(
    text_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    limit: int = 10,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion — merge two ranked result lists.

    score(doc) = 1/(k + rank_text) + 1/(k + rank_vector)
    where k=60 is the standard RRF constant.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(text_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = doc

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in docs:
            docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{**docs[doc_id], "rrf_score": score} for doc_id, score in ranked]


async def get_provenance_chain(db: UCDatabase, artifact_id: str) -> list[dict[str, Any]]:
    """Trace the full provenance chain for an artifact back to its scope."""
    return await db.query(
        f"SELECT <-produced<-turn<-contains<-run<-contains<-scope FROM {artifact_id}"
    )


async def get_artifact_lineage(db: UCDatabase, artifact_id: str) -> list[dict[str, Any]]:
    """Get all artifacts this one depends on."""
    return await db.query(f"SELECT ->depends_on->artifact FROM {artifact_id}")


async def get_turn_artifacts(db: UCDatabase, turn_id: str) -> list[dict[str, Any]]:
    """Get all artifacts produced by a turn."""
    return await db.query(f"SELECT ->produced->artifact FROM {turn_id}")


async def get_turn_summaries(db: UCDatabase, run_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get turns for a run with their summary artifacts.

    For each turn, traverses turn->produced->artifact to find kind='summary'
    artifacts. Returns list of {turn_id, sequence, user_message, summary, started_at}.
    """
    turns = await db.query(
        f"SELECT * FROM turn WHERE run = {run_id} ORDER BY sequence DESC LIMIT {limit}"
    )

    results = []
    for t in turns:
        tid = str(t["id"])
        # Walk the graph: turn->produced->transcript<-depends_on<-summary
        # First find transcript artifacts for this turn
        artifacts = await db.query(f"SELECT ->produced->artifact FROM {tid}")
        summary_text = None
        for a in artifacts:
            produced = a.get("->produced", {})
            if isinstance(produced, dict):
                for aid in produced.get("->artifact", []):
                    aid_str = str(aid)
                    # Reverse-traverse: find artifacts that depend on this one
                    rev = await db.query(f"SELECT <-depends_on<-artifact FROM {aid_str}")
                    for row in rev:
                        dep = row.get("<-depends_on", {})
                        if isinstance(dep, dict):
                            dep_ids = dep.get("<-artifact", [])
                            for did in dep_ids:
                                did_str = str(did)
                                detail = await db.query(f"SELECT kind, content FROM {did_str}")
                                if detail and detail[0].get("kind") == "summary":
                                    summary_text = detail[0].get("content", "")
                                    break
                        if summary_text is not None:
                            break
            if summary_text is not None:
                break

        results.append(
            {
                "turn_id": tid,
                "sequence": t.get("sequence"),
                "user_message": t.get("user_message"),
                "summary": summary_text,
                "started_at": t.get("started_at"),
            }
        )

    return results


# ============================================================
# JOB QUEUE
# ============================================================


async def create_job(
    db: UCDatabase,
    job_type: str,
    target: str,
    priority: int = 0,
) -> str:
    """Create a new job. Returns the job ID string."""
    jid = _gen_id()
    await db.query(
        f'CREATE job:{jid} SET job_type = $job_type, status = "pending", '
        f"target = $target, priority = $priority",
        {"job_type": job_type, "target": target, "priority": priority},
    )
    return f"job:{jid}"


async def claim_next_job(db: UCDatabase) -> dict[str, Any] | None:
    """Claim the next pending job (highest priority, oldest first).

    Uses plain SELECT + UPDATE (two queries).  This is safe because the
    daemon runs a single serial worker — no concurrent claim risk.

    Note: SurrealDB Python SDK v1 returns ``[]`` for ``BEGIN TRANSACTION``
    blocks, so the transaction-based approach silently swallowed results,
    causing jobs to leak into "running" without ever being processed.
    """
    # Retry a few times in case another worker/daemon races us on the same candidate.
    for _ in range(3):
        candidates = await db.query(
            'SELECT * FROM job WHERE status = "pending" '
            "ORDER BY priority DESC, created_at ASC LIMIT 1"
        )
        if not candidates:
            return None
        job_id = str(candidates[0]["id"])
        result = await db.query(
            f'UPDATE {job_id} SET status = "running", started_at = time::now() '
            'WHERE status = "pending"'
        )
        if result:
            return result[0]
    return None


async def complete_job(db: UCDatabase, job_id: str, result: dict[str, Any] | None = None) -> None:
    """Mark a job as completed."""
    await db.query(
        f'UPDATE {job_id} SET status = "completed", completed_at = time::now(), result = $result',
        {"result": result or {}},
    )


async def fail_job(db: UCDatabase, job_id: str, error: str) -> None:
    """Mark a job as failed. Increments attempts; resets to pending if retries remain."""
    # Read current state
    current = await db.query(f"SELECT attempts, max_attempts FROM {job_id}")
    if not current:
        return

    attempts = current[0].get("attempts", 0) + 1
    max_attempts = current[0].get("max_attempts", 10)
    new_status = "failed" if attempts >= max_attempts else "pending"

    await db.query(
        f"UPDATE {job_id} SET attempts = $attempts, error = $error, "
        f"status = $status, started_at = NONE",
        {"attempts": attempts, "error": error, "status": new_status},
    )


async def list_jobs(
    db: UCDatabase,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List jobs, optionally filtered."""
    conditions = []
    params: dict[str, Any] = {}
    if status:
        conditions.append("status = $status")
        params["status"] = status
    if job_type:
        conditions.append("job_type = $job_type")
        params["job_type"] = job_type

    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    return await db.query(
        f"SELECT * FROM job{where} ORDER BY created_at DESC LIMIT {limit}", params
    )


async def count_jobs_by_status(db: UCDatabase) -> dict[str, int]:
    """Get job counts grouped by status."""
    result = await db.query("SELECT status, count() FROM job GROUP BY status")
    return {row["status"]: row["count"] for row in result} if result else {}


async def recover_stale_running_jobs(db: UCDatabase) -> int:
    """Reset zombie running jobs back to pending.

    On daemon restart, any jobs stuck in 'running' status were abandoned
    mid-processing by the dead worker.  Reset them so the new worker can
    reclaim them.
    """
    result = await db.query(
        'UPDATE job SET status = "pending", started_at = NONE '
        'WHERE status = "running"'
    )
    return len(result) if isinstance(result, list) else 0


async def prune_stale_pending_jobs(db: UCDatabase) -> int:
    """Remove pending turn_summary jobs for turns that already have summaries.

    After a reimport, pending jobs from the export snapshot remain but their
    target turns may already have summary artifacts.  This prevents duplicate
    summaries and wasted LLM calls.

    Uses Python iteration with simple queries to stay compatible with both
    embedded (v2) and server (v3) modes.
    """
    jobs = await db.query(
        'SELECT * FROM job WHERE status = "pending" AND job_type = "turn_summary"'
    )
    if not isinstance(jobs, list) or not jobs:
        return 0

    pruned = 0
    for job in jobs:
        target = str(job.get("target", ""))
        if not target:
            continue
        # target is a string like "turn:abc123"
        # Check: does this turn have transcript artifacts that a summary depends on?
        artifacts = await db.query(f"SELECT out FROM produced WHERE in = {target}")
        if not isinstance(artifacts, list):
            continue

        has_summary = False
        for edge in artifacts:
            artifact_id = edge.get("out")
            if not artifact_id:
                continue
            aid_str = str(artifact_id)
            # Find artifacts that depend on this transcript (in=dependent, out=source)
            deps = await db.query(f"SELECT in FROM depends_on WHERE out = {aid_str}")
            for dep in deps if isinstance(deps, list) else []:
                dep_id = dep.get("in")
                if not dep_id:
                    continue
                # Verify the dependent is actually a summary artifact
                dep_detail = await db.query(
                    f"SELECT kind FROM {str(dep_id)}"
                )
                if dep_detail and dep_detail[0].get("kind") == "summary":
                    has_summary = True
                    break
            if has_summary:
                break

        if not has_summary:
            continue

        job_id = str(job["id"])
        await db.query(f"DELETE {job_id}")
        pruned += 1

    return pruned


# ============================================================
# WORKING MEMORY
# ============================================================


async def get_working_memory(
    db: UCDatabase,
    scope_id: str,
) -> dict[str, Any] | None:
    """Get the most recent working memory artifact for a scope.

    Working memory artifacts have kind='working_memory' and
    metadata.scope_id pointing to the scope they belong to.
    """
    result = await db.query(
        'SELECT * FROM artifact WHERE kind = "working_memory" '
        "AND metadata.scope_id = $scope_id "
        "ORDER BY created_at DESC LIMIT 1",
        {"scope_id": scope_id},
    )
    return result[0] if result else None


async def upsert_working_memory(
    db: UCDatabase,
    scope_id: str,
    content: str,
    method: str = "llm",
) -> str:
    """Create a new working memory artifact for a scope.

    Artifacts are immutable (append-only). If a previous version exists,
    creates a depends_on edge with 'supersedes' relationship.
    Returns the new artifact ID string.
    """
    # Find previous version
    previous = await get_working_memory(db, scope_id)

    aid = _gen_id()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    # Inline scope reference — scope_id is e.g. "scope:abc123"
    scope_clause = f", scope = {scope_id}" if scope_id.startswith("scope:") else ""
    await db.query(
        f"CREATE artifact:{aid} SET kind = $kind, content = $content, "
        f"content_hash = $content_hash, metadata = $metadata{scope_clause}",
        {
            "kind": "working_memory",
            "content": content,
            "content_hash": content_hash,
            "metadata": {"scope_id": scope_id, "method": method},
        },
    )

    new_id = f"artifact:{aid}"

    # Link to previous version via supersedes edge
    if previous:
        prev_id = str(previous["id"])
        await db.query(
            f"RELATE {new_id}->depends_on->{prev_id} SET relationship = $rel",
            {"rel": "supersedes"},
        )

    return new_id


async def get_working_memory_history(
    db: UCDatabase,
    scope_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get version history of working memory artifacts for a scope."""
    return await db.query(
        'SELECT * FROM artifact WHERE kind = "working_memory" '
        "AND metadata.scope_id = $scope_id "
        f"ORDER BY created_at DESC LIMIT {limit}",
        {"scope_id": scope_id},
    )


async def set_working_memory_reasoning_metadata(
    db: UCDatabase,
    scope_id: str,
    reasoning: dict[str, Any],
) -> str | None:
    """Attach latest deep-reasoning snapshot metadata to current working memory.

    Returns the working memory artifact ID if updated, otherwise None.
    """
    memory = await get_working_memory(db, scope_id)
    if not memory:
        return None

    mid = str(memory["id"])
    await db.query(
        f"UPDATE {mid} SET metadata.last_reasoning = $reasoning, "
        "metadata.last_reasoning_at = time::now()",
        {"reasoning": reasoning},
    )
    return mid


async def get_scope_summaries_for_distillation(
    db: UCDatabase,
    scope_id: str,
    limit: int = 30,
    branch: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent turn summaries across all runs in a scope.

    Walks: scope -> runs -> turns -> produced -> transcript <- depends_on <- summary.
    Returns a flat list of dicts with run context for LLM distillation.
    """
    # Get runs for scope, newest first, optionally filtered by branch
    branch_filter = " AND branch = $branch" if branch else ""
    runs = await db.query(
        f"SELECT * FROM run WHERE scope = {scope_id}{branch_filter} ORDER BY started_at DESC",
        {"branch": branch} if branch else {},
    )

    results: list[dict[str, Any]] = []
    for run in runs:
        if len(results) >= limit:
            break

        rid = str(run["id"])
        agent_type = run.get("agent_type", "")
        summaries = await get_turn_summaries(db, rid, limit=limit - len(results))

        for s in summaries:
            results.append(
                {
                    "run_id": rid,
                    "agent_type": agent_type,
                    "sequence": s.get("sequence"),
                    "user_message": s.get("user_message"),
                    "summary": s.get("summary"),
                    "started_at": s.get("started_at"),
                }
            )

    return results


async def backfill_artifact_scopes(db: UCDatabase, batch_size: int = 100) -> int:
    """Backfill scope field on existing artifacts that lack it.

    Walks: artifact <-produced<- turn -> run -> scope
    For working_memory artifacts, uses metadata.scope_id instead.
    Processes in batches to avoid overwhelming the KV store.
    Returns the number of artifacts updated.
    """
    import logging

    logger = logging.getLogger(__name__)
    count = 0

    # 1. Transcript/summary artifacts via provenance graph
    orphans = await db.query(
        f"SELECT id FROM artifact WHERE scope = NONE AND kind != 'working_memory' "
        f"LIMIT {batch_size}"
    )
    if not isinstance(orphans, list):
        logger.warning("Backfill query returned non-list: %s", str(orphans)[:200])
        orphans = []

    for orphan in orphans:
        if not isinstance(orphan, dict):
            continue
        oid = str(orphan["id"])
        # Walk: artifact <-produced<- turn.run.scope
        chain = await db.query(f"SELECT <-produced<-turn.run.scope AS scope_id FROM {oid}")
        scope_ref = _extract_scope_from_traversal(chain if isinstance(chain, list) else [])

        # For derived artifacts, walk depends_on -> source -> produced -> turn
        if scope_ref is None:
            dep_chain = await db.query(
                f"SELECT ->depends_on->artifact<-produced<-turn.run.scope AS scope_id FROM {oid}"
            )
            scope_ref = _extract_scope_from_traversal(
                dep_chain if isinstance(dep_chain, list) else []
            )

        if scope_ref and scope_ref.startswith("scope:"):
            await db.query(f"UPDATE {oid} SET scope = {scope_ref}")
            count += 1

    # 2. Working memory artifacts via metadata.scope_id
    wm_orphans = await db.query(
        "SELECT id, metadata FROM artifact "
        f"WHERE scope = NONE AND kind = 'working_memory' LIMIT {batch_size}"
    )
    if not isinstance(wm_orphans, list):
        logger.warning("WM backfill query returned non-list: %s", str(wm_orphans)[:200])
        wm_orphans = []

    for wm in wm_orphans:
        if not isinstance(wm, dict):
            continue
        wm_id = str(wm["id"])
        meta_scope = (wm.get("metadata") or {}).get("scope_id", "")
        if meta_scope and str(meta_scope).startswith("scope:"):
            await db.query(f"UPDATE {wm_id} SET scope = {meta_scope}")
            count += 1

    return count


async def backfill_canonical_ids(db: UCDatabase) -> int:
    """Backfill canonical_id on scopes that lack it.

    For each scope with a valid filesystem path:
    1. Compute resolve_canonical_id(path)
    2. If another scope already has that canonical_id → merge into it
    3. Otherwise → set the canonical_id

    Returns the number of scopes updated or merged.
    """
    import logging
    from pathlib import Path

    from ..git import resolve_canonical_id

    logger = logging.getLogger(__name__)
    count = 0

    scopes = await db.query("SELECT * FROM scope WHERE !canonical_id")
    if not isinstance(scopes, list):
        return 0

    for scope in scopes:
        path = scope.get("path")
        if not path:
            continue

        scope_id = str(scope["id"])
        try:
            canonical_id = resolve_canonical_id(Path(path))
        except Exception:
            logger.debug("Failed to resolve canonical_id for %s", path)
            continue

        # Check if another scope already owns this canonical_id
        existing = await find_scope_by_canonical_id(db, canonical_id)
        if existing and str(existing["id"]) != scope_id:
            # Merge this scope into the existing one
            logger.info(
                "Merging scope %s into %s (shared canonical_id: %s)",
                scope_id,
                str(existing["id"]),
                canonical_id,
            )
            await merge_scopes(db, scope_id, str(existing["id"]))
            count += 1
        else:
            await update_scope(db, scope_id, canonical_id=canonical_id)
            count += 1

    return count


async def detect_merged_runs(
    db: UCDatabase,
    scope_id: str,
    current_branch: str,
    repo_path: Any,
) -> int:
    """Tag runs from other branches whose commits are ancestors of current_branch.

    For each run on a different branch with a commit_sha, checks whether that
    commit has been merged into current_branch using git merge-base --is-ancestor.
    Sets merged_to on matching runs.

    Returns the number of runs tagged.
    """
    import asyncio
    from pathlib import Path

    from ..git import is_ancestor

    runs = await db.query(
        f"SELECT * FROM run WHERE scope = {scope_id} "
        "AND branch != $branch AND !!branch "
        "AND !!commit_sha AND !merged_to",
        {"branch": current_branch},
    )
    if not isinstance(runs, list):
        return 0

    count = 0
    for run in runs:
        commit = run.get("commit_sha")
        if not commit:
            continue
        ancestor = await asyncio.to_thread(
            is_ancestor,
            Path(str(repo_path)),
            commit,
            current_branch,
        )
        if ancestor:
            run_id = str(run["id"])
            await db.query(
                f"UPDATE {run_id} SET merged_to = $branch",
                {"branch": current_branch},
            )
            count += 1
    return count


def _extract_scope_from_traversal(chain: list[dict[str, Any]]) -> str | None:
    """Extract a scope:xxx reference from SurrealDB graph traversal results.

    Graph traversals return nested lists — this unwraps them to find
    the first non-None scope reference.
    """
    if not chain:
        return None
    scope_id_val = chain[0].get("scope_id")
    if scope_id_val is None:
        return None
    if isinstance(scope_id_val, list):
        for item in scope_id_val:
            if isinstance(item, list):
                for inner in item:
                    if inner is not None:
                        return str(inner)
            elif item is not None:
                return str(item)
        return None
    return str(scope_id_val)
