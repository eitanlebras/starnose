"""snose inspect — TUI for inspecting a run's context window."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TextLog,
)

from starnose.db import Database, Run
from starnose.tokens import get_context_limit


ROLE_COLORS = {
    "system": "#4488ff",
    "user": "#00cc66",
    "assistant": "#ffdd44",
    "tool": "#aa66ff",
}

CSS = """
Screen {
    background: #0d0d0d;
}

#left-panel {
    width: 40%;
    border: solid #2a2a2a;
    padding: 1;
}

#right-panel {
    width: 60%;
    border: solid #2a2a2a;
    padding: 1;
}

.panel-title {
    text-style: bold;
    color: #e8e8e8;
    margin-bottom: 1;
}

.meta-label {
    color: #666666;
}

.budget-bar {
    margin: 1 0;
}

.warning {
    color: #ffaa00;
}

.segment-table {
    height: auto;
    max-height: 12;
}

.call-list {
    height: 1fr;
}

#message-view {
    height: 1fr;
}

DataTable {
    height: auto;
}
"""


class InspectApp(App):
    """TUI for inspecting a single run."""

    TITLE = "snose inspect"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("e", "export", "Export"),
        Binding("d", "diff", "Diff"),
        Binding("question_mark", "help", "Help"),
        Binding("slash", "search", "Search"),
    ]

    def __init__(self, run: Run, db: Database):
        super().__init__()
        self.run_data = run
        self.db = db
        self._selected_call_idx = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left-panel"):
                yield Static(self._build_meta(), id="meta")
                yield Static(self._build_budget_bar(), id="budget", classes="budget-bar")
                yield Static(self._build_segment_breakdown(), id="segments")
                yield DataTable(id="call-table", classes="call-list")
            with VerticalScroll(id="right-panel"):
                yield Static("[bold]Messages[/]", classes="panel-title")
                yield Static(id="message-detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#call-table", DataTable)
        table.add_columns("#", "Model", "Input", "Output", "Latency")
        table.cursor_type = "row"

        for call in self.run_data.calls:
            table.add_row(
                str(call.sequence),
                call.model or "—",
                f"{call.input_tokens:,}",
                f"{call.output_tokens:,}",
                f"{call.latency_ms:,}ms",
            )

        if self.run_data.calls:
            self._show_call(0)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._selected_call_idx = event.cursor_row
        self._show_call(event.cursor_row)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._selected_call_idx = event.cursor_row
        self._show_call(event.cursor_row)

    def _show_call(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.run_data.calls):
            return
        call = self.run_data.calls[idx]
        lines = []
        for seg in call.segments:
            color = ROLE_COLORS.get(seg.role, "#e8e8e8")
            badge = f"[{color} bold]{seg.role.upper()}[/]"
            type_label = f"[dim]{seg.seg_type}[/]"
            tok_label = f"[dim]{seg.token_count:,} tok[/]"

            # Warning for large tool results
            warning = ""
            if seg.seg_type == "tool_result" and seg.token_count > 2000:
                warning = " [#ffaa00]⚠ large tool result[/]"

            preview = seg.content[:200].replace("\n", "↵ ") if seg.content else ""
            lines.append(f"{badge}  {type_label}  {tok_label}{warning}")
            lines.append(f"  [dim]{preview}[/]")
            lines.append("")

        try:
            detail = self.query_one("#message-detail", Static)
            detail.update("\n".join(lines) if lines else "[dim]No messages[/]")
        except NoMatches:
            pass

    def _build_meta(self) -> str:
        r = self.run_data
        total_tokens = sum(c.input_tokens + c.output_tokens for c in r.calls)
        duration = "—"
        if r.started_at and r.finished_at:
            secs = (r.finished_at - r.started_at).total_seconds()
            duration = f"{secs:.1f}s" if secs < 60 else f"{secs / 60:.1f}m"

        status_icon = {"success": "✓", "failed": "✗", "running": "●"}.get(r.status, r.status)

        lines = [
            f"[bold]{r.name or r.id}[/]",
            f"[dim]ID:[/] {r.id}  [dim]Status:[/] {status_icon} {r.status}",
            f"[dim]Duration:[/] {duration}  [dim]Calls:[/] {len(r.calls)}  [dim]Tokens:[/] {total_tokens:,}",
        ]
        return "\n".join(lines)

    def _build_budget_bar(self) -> str:
        r = self.run_data
        total_tokens = sum(c.input_tokens + c.output_tokens for c in r.calls)
        model = r.calls[0].model if r.calls else "gpt-4o"
        ctx_limit = get_context_limit(model)
        pct = total_tokens / ctx_limit * 100 if ctx_limit else 0
        filled = int(pct / 5)
        bar = "▓" * filled + "░" * (20 - filled)

        color = "#00d4aa"
        warning = ""
        if pct > 80:
            color = "#ffaa00"
            warning = " [#ffaa00]⚠ context budget high[/]"
        if pct > 95:
            color = "#ff4455"

        return f"[{color}]{bar}[/]  {pct:.0f}%  {total_tokens:,} / {ctx_limit:,} tok{warning}"

    def _build_segment_breakdown(self) -> str:
        seg_totals: dict[str, int] = {}
        for call in self.run_data.calls:
            for seg in call.segments:
                if seg.seg_type != "output":
                    seg_totals[seg.seg_type] = seg_totals.get(seg.seg_type, 0) + seg.token_count

        if not seg_totals:
            return "[dim]No segments[/]"

        total = sum(seg_totals.values())
        lines = ["[bold]Segment Breakdown[/]", ""]

        for seg_type, count in sorted(seg_totals.items(), key=lambda x: -x[1]):
            share = count / total * 100 if total else 0
            bar_len = int(share / 10)
            bar = "▓" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  {seg_type:16} {count:>6,}  [#00d4aa]{bar}[/]  {share:.0f}%")

        return "\n".join(lines)

    def action_export(self) -> None:
        """Export run to JSON."""
        import json
        from starnose.cli import export
        self.notify(f"Export: snose export {self.run_data.id}")

    def action_diff(self) -> None:
        self.notify("Use: snose diff from the command line")

    def action_help(self) -> None:
        self.notify(
            "↑↓ navigate | Enter expand | / search | d diff | e export | q quit"
        )
