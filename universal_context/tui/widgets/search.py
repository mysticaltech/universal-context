"""Search pane — full-text search across artifacts."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Input, Static


class SearchPane(Static):
    """Search widget with query input and results table."""

    def __init__(self, db_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._db_path = db_path

    def compose(self) -> ComposeResult:
        with Vertical(classes="pane-container"):
            yield Input(
                placeholder="Search artifacts...",
                id="search-input",
                classes="search-input",
            )
            yield DataTable(id="search-results", classes="data-table")
            yield Static("", id="artifact-detail", classes="detail-view")

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.add_columns("ID", "Kind", "Content Preview")
        table.cursor_type = "row"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run search when Enter is pressed."""
        query = event.value.strip()
        if query:
            self.run_worker(self._do_search(query))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show artifact detail when a row is selected."""
        table = self.query_one("#search-results", DataTable)
        row = table.get_row(event.row_key)
        detail = self.query_one("#artifact-detail", Static)
        if row:
            detail.update(
                f"[bold]{row[0]}[/bold] ({row[1]})\n\n{row[2]}"
            )

    async def _do_search(self, query: str) -> None:
        from ...db.queries import search_artifacts
        from ...db.schema import apply_schema

        db = self._make_db()
        await db.connect()
        try:
            await apply_schema(db)
            results = await search_artifacts(db, query, limit=20)

            table = self.query_one("#search-results", DataTable)
            table.clear()

            if not results:
                status = self.query_one("#artifact-detail", Static)
                status.update("[dim]No results found.[/dim]")
                return

            for r in results:
                content = (r.get("content") or "")[:80]
                table.add_row(
                    str(r["id"]),
                    r.get("kind", ""),
                    content,
                )
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
