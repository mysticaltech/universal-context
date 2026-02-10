"""Overview pane — shows scopes, recent runs, and job queue status."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static


class OverviewPane(Static):
    """Dashboard overview: scopes, runs, jobs."""

    def __init__(self, db_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._db_path = db_path

    def compose(self) -> ComposeResult:
        with Vertical(classes="pane-container"):
            yield Static("[bold]Scopes[/bold]", id="scopes-header")
            yield DataTable(id="scopes-table", classes="data-table")
            yield Static("[bold]Recent Runs[/bold]", id="runs-header")
            yield DataTable(id="runs-table", classes="data-table")
            yield Static("[bold]Job Queue[/bold]", id="jobs-header")
            yield Static("", id="jobs-status")

    def on_mount(self) -> None:
        scopes_table = self.query_one("#scopes-table", DataTable)
        scopes_table.add_columns("ID", "Name", "Path")

        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.add_columns("ID", "Agent", "Status", "Started")

        self.refresh_data()

    def refresh_data(self) -> None:
        self.run_worker(self._load_data())

    async def _load_data(self) -> None:
        from ...db.queries import count_jobs_by_status, list_runs, list_scopes
        from ...db.schema import apply_schema

        db = self._make_db()
        await db.connect()
        try:
            await apply_schema(db)

            # Scopes
            scopes = await list_scopes(db)
            scopes_table = self.query_one("#scopes-table", DataTable)
            scopes_table.clear()
            for s in scopes[:10]:
                scopes_table.add_row(
                    str(s["id"]), s.get("name", ""), s.get("path", "")
                )

            # Runs
            runs = await list_runs(db, limit=10)
            runs_table = self.query_one("#runs-table", DataTable)
            runs_table.clear()
            for r in runs:
                runs_table.add_row(
                    str(r["id"]),
                    r.get("agent_type", ""),
                    r.get("status", ""),
                    str(r.get("started_at", ""))[:19],
                )

            # Jobs
            jobs = await count_jobs_by_status(db)
            jobs_widget = self.query_one("#jobs-status", Static)
            if jobs:
                parts = [f"{status}: {count}" for status, count in jobs.items()]
                jobs_widget.update("  ".join(parts))
            else:
                jobs_widget.update("[dim]No jobs[/dim]")
        finally:
            await db.close()

    def _make_db(self):
        from ...db.client import UCDatabase

        if self._db_path:
            return UCDatabase.from_path(Path(self._db_path))

        from ...config import UCConfig

        config = UCConfig.load()
        if config.db_url:
            return UCDatabase.from_url(config.db_url, config.db_user, config.db_pass)
        return UCDatabase.from_path(Path(config.resolved_db_path))
