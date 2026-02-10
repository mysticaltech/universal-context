"""UC Dashboard — Textual TUI application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .widgets.overview import OverviewPane
from .widgets.search import SearchPane
from .widgets.timeline import TimelinePane


class UCDashboard(App):
    """Universal Context TUI Dashboard."""

    TITLE = "Universal Context"
    CSS = """
    Screen {
        background: $surface;
    }

    TabbedContent {
        height: 1fr;
    }

    #status-bar {
        height: 1;
        dock: bottom;
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    .pane-container {
        height: 1fr;
        padding: 1;
    }

    .data-table {
        height: 1fr;
    }

    .search-input {
        dock: top;
        margin-bottom: 1;
    }

    .detail-view {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "tab_overview", "Overview", show=False),
        Binding("2", "tab_timeline", "Timeline", show=False),
        Binding("3", "tab_search", "Search", show=False),
    ]

    def __init__(self, db_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._db_path = db_path

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Overview", id="tab-overview"):
                yield OverviewPane(db_path=self._db_path)
            with TabPane("Timeline", id="tab-timeline"):
                yield TimelinePane(db_path=self._db_path)
            with TabPane("Search", id="tab-search"):
                yield SearchPane(db_path=self._db_path)
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh all panes."""
        for pane in self.query(OverviewPane):
            pane.refresh_data()
        for pane in self.query(TimelinePane):
            pane.refresh_data()

    def action_tab_overview(self) -> None:
        self.query_one(TabbedContent).active = "tab-overview"

    def action_tab_timeline(self) -> None:
        self.query_one(TabbedContent).active = "tab-timeline"

    def action_tab_search(self) -> None:
        self.query_one(TabbedContent).active = "tab-search"


def run_dashboard(db_path: str | None = None) -> None:
    """Launch the TUI dashboard."""
    app = UCDashboard(db_path=db_path)
    app.run()
