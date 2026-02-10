"""Timeline pane — shows turns for a selected run."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static


class TimelinePane(Static):
    """Chronological view of turns in a run."""

    def __init__(self, db_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._db_path = db_path
        self._current_run_id: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="pane-container"):
            yield Static("[bold]Timeline[/bold] (latest run)", id="timeline-header")
            yield DataTable(id="timeline-table", classes="data-table")
            yield Static("", id="turn-detail", classes="detail-view")

    def on_mount(self) -> None:
        table = self.query_one("#timeline-table", DataTable)
        table.add_columns("#", "User Message", "Started")
        table.cursor_type = "row"
        self.refresh_data()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show turn details when a row is selected."""
        table = self.query_one("#timeline-table", DataTable)
        row = table.get_row(event.row_key)
        detail = self.query_one("#turn-detail", Static)
        if row:
            detail.update(
                f"[bold]Turn {row[0]}[/bold]\n"
                f"User: {row[1]}\n"
                f"Started: {row[2]}"
            )

    def refresh_data(self) -> None:
        self.run_worker(self._load_data())

    async def _load_data(self) -> None:
        from ...db.queries import list_runs, list_turns
        from ...db.schema import apply_schema

        db = self._make_db()
        await db.connect()
        try:
            await apply_schema(db)

            # Get latest run
            runs = await list_runs(db, limit=1)
            if not runs:
                header = self.query_one("#timeline-header", Static)
                header.update("[bold]Timeline[/bold] — no runs found")
                return

            run = runs[0]
            rid = str(run["id"])
            self._current_run_id = rid

            header = self.query_one("#timeline-header", Static)
            header.update(
                f"[bold]Timeline[/bold] — {rid} "
                f"({run.get('agent_type', '')}, {run.get('status', '')})"
            )

            turns = await list_turns(db, rid)
            table = self.query_one("#timeline-table", DataTable)
            table.clear()
            for t in turns:
                msg = (t.get("user_message") or "")[:50]
                ts = str(t.get("started_at", ""))[:19]
                table.add_row(str(t.get("sequence", "")), msg, ts)
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
