"""Core domain models for Universal Context.

These are product-level concepts, not database schemas.
SurrealDB records are created from these models but the models remain DB-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# --- Enums ---


class RunStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CRASHED = "crashed"


class AgentType(StrEnum):
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


class ArtifactKind(StrEnum):
    TRANSCRIPT = "transcript"
    DIFF = "diff"
    PATCH = "patch"
    FILE = "file"
    LOG = "log"
    SUMMARY = "summary"
    NOTE = "note"
    IMAGE = "image"
    WORKING_MEMORY = "working_memory"


class StepAction(StrEnum):
    TOOL_CALL = "tool_call"
    FILE_WRITE = "file_write"
    COMMAND = "command"
    API_CALL = "api_call"


class JobType(StrEnum):
    TURN_SUMMARY = "turn_summary"
    SESSION_SUMMARY = "session_summary"
    EMBEDDING = "embedding"
    EXTRACTION = "extraction"
    MEMORY_UPDATE = "memory_update"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProvenanceRelation(StrEnum):
    DERIVED_FROM = "derived_from"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"


# --- Core Models ---


class Scope(BaseModel):
    """A named container for work — project, repo, investigation, or workspace."""

    id: str | None = None
    name: str
    path: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    """A single agent session from start to finish."""

    id: str | None = None
    scope_id: str
    agent_type: AgentType
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: RunStatus = RunStatus.ACTIVE
    session_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """One user -> agent -> response cycle."""

    id: str | None = None
    run_id: str
    sequence: int
    started_at: datetime | None = None
    ended_at: datetime | None = None
    user_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """A concrete action inside a turn (tool call, file write, command, API call)."""

    id: str | None = None
    turn_id: str
    sequence: int
    action_type: StepAction
    action_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None


class Artifact(BaseModel):
    """Any produced or referenced object. Immutable once recorded."""

    id: str | None = None
    kind: ArtifactKind
    content: str | None = None
    content_hash: str | None = None
    blob_path: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """A resumable state snapshot of a run at a specific turn."""

    id: str | None = None
    run_id: str
    turn_id: str
    label: str | None = None
    created_at: datetime | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    """An asynchronous processing task (summarization, embedding, extraction)."""

    id: str | None = None
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    target_id: str  # record ID of the target (run, turn, artifact)
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 10
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class TurnInfo(BaseModel):
    """Information about a detected turn, returned by triggers."""

    sequence: int
    user_message: str | None = None
    assistant_response: str | None = None
    raw_content: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
