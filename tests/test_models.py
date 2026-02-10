"""Tests for domain models."""

from universal_context.models.types import (
    Artifact,
    ArtifactKind,
    Checkpoint,
    Job,
    JobStatus,
    JobType,
    Run,
    RunStatus,
    Scope,
    Step,
    StepAction,
    TurnInfo,
)


def test_scope_creation():
    scope = Scope(name="my-project", path="/home/user/project")
    assert scope.name == "my-project"
    assert scope.path == "/home/user/project"
    assert scope.metadata == {}


def test_run_defaults():
    run = Run(scope_id="scope:abc", agent_type="claude")
    assert run.status == RunStatus.ACTIVE
    assert run.ended_at is None
    assert run.metadata == {}


def test_artifact_immutable_fields():
    artifact = Artifact(kind=ArtifactKind.TRANSCRIPT, content="hello world")
    assert artifact.kind == "transcript"
    assert artifact.content == "hello world"
    assert artifact.blob_path is None


def test_job_defaults():
    job = Job(job_type=JobType.TURN_SUMMARY, target_id="turn:123")
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0
    assert job.max_attempts == 10
    assert job.priority == 0


def test_turn_info():
    info = TurnInfo(
        sequence=1,
        user_message="fix the bug",
        tool_calls=[{"name": "read_file", "args": {"path": "main.py"}}],
        files_changed=["main.py"],
    )
    assert info.sequence == 1
    assert len(info.tool_calls) == 1
    assert info.files_changed == ["main.py"]


def test_step_action_enum():
    step = Step(turn_id="turn:1", sequence=0, action_type=StepAction.TOOL_CALL)
    assert step.action_type == "tool_call"


def test_checkpoint():
    cp = Checkpoint(run_id="run:1", turn_id="turn:5", label="before-refactor")
    assert cp.label == "before-refactor"
    assert cp.state == {}
