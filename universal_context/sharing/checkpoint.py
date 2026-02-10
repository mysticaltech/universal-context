"""Checkpoint creation and restoration.

A checkpoint captures the state of a run at a specific turn,
allowing later restoration or branching from that point.
"""

from __future__ import annotations

from typing import Any

from ..db.client import UCDatabase
from ..db.queries import get_run, get_turn, list_turns


async def create_checkpoint(
    db: UCDatabase,
    run_id: str,
    turn_id: str,
    label: str | None = None,
) -> str:
    """Create a checkpoint at a specific turn in a run.

    Captures turn count, run metadata, and turn sequence position.
    Returns the checkpoint ID.
    """
    run = await get_run(db, run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    turn = await get_turn(db, turn_id)
    if not turn:
        raise ValueError(f"Turn not found: {turn_id}")

    turns = await list_turns(db, run_id)
    turn_count = len(turns)

    from ..db.queries import _gen_id

    cid = _gen_id()
    state = {
        "run_id": run_id,
        "turn_id": turn_id,
        "turn_sequence": turn.get("sequence", 0),
        "total_turns": turn_count,
        "agent_type": run.get("agent_type", ""),
        "status": run.get("status", ""),
    }

    await db.query(
        f"CREATE checkpoint:{cid} SET run = {run_id}, turn = {turn_id}, "
        f"label = $label, state = $state",
        {"label": label, "state": state},
    )

    # Create provenance edge
    await db.query(f"RELATE {run_id}->checkpoint_at->checkpoint:{cid}")

    return f"checkpoint:{cid}"


async def list_checkpoints(
    db: UCDatabase, run_id: str | None = None
) -> list[dict[str, Any]]:
    """List checkpoints, optionally filtered by run."""
    if run_id:
        return await db.query(
            f"SELECT * FROM checkpoint WHERE run = {run_id} "
            "ORDER BY created_at DESC"
        )
    return await db.query("SELECT * FROM checkpoint ORDER BY created_at DESC")


async def get_checkpoint(
    db: UCDatabase, checkpoint_id: str
) -> dict[str, Any] | None:
    """Get a checkpoint by ID."""
    result = await db.query(f"SELECT * FROM {checkpoint_id}")
    return result[0] if result else None
