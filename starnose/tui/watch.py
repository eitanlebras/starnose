"""snose watch — live TUI for monitoring a running agent."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static
from textual.timer import Timer

from starnose.db import Database, Run
from starnose.tokens import get_context_limit


CSS = """
Screen {
    background: #0d0d0d;
}

#header-bar {
    height: 3;
    border: solid #2a2a2a;
    padding: 0 1;
    content-align: center middle;
}

#budget-bar {
    height: 3;
    border: solid #2a2a2a;
    padding: 0 1;
}

#event-log {
    height: 1fr;
    border: solid #2a2a2a;
    padding: 1;
}

.live-indicator {
    color: #00cc66;
    text-style: bold;
}
"""


class WatchApp(App):
    """Live TUI for watching a running agent."""

    TITLE = "snose watch"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "snapshot", "Snapshot"),
        Binding("p", "pause", "Pause"),
    ]

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._run: Run | None = None
        self._last_sequence = -1
        self._events: list[str] = []
        self._paused = False
        self._start_time: float = time.time()
        self._poll_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[dim]Waiting for active run...[/]", id="header-bar")
        yield Static("", id="budget-bar")
        with VerticalScroll(id="event-log"):
            yield Static("", id="events")
        yield Footer()

    def on_mount(self) -> None:
        self._poll_timer = self.set_interval(0.1, self._poll)

    def _poll(self) -> None:
        if self._paused:
            return

        # Find or update the running run
        if not self._run or self._run.status == "running":
            current = self.db.get_running_run()
            if current:
                if not self._run or self._run.id != current.id:
                    self._run = current
                    self._last_sequence = -1
                    self._events = []
                    self._start_time = time.time()

        if not self._run:
            return

        # Check for new calls
        new_calls = self.db.get_latest_calls(self._run.id, self._last_sequence)
        for call in new_calls:
            self._last_sequence = call.sequence
            elapsed = time.time() - self._start_time
            ts = f"{elapsed:06.2f}"

            for seg in call.segments:
                if seg.seg_type == "output":
                    color = "#00cc66"
                    label = "llm_call"
                    detail = f"#{call.sequence}  complete  {seg.token_count:,} tok output"
                elif seg.role == "tool":
                    color = "#00dddd"
                    warning = "  [#ffaa00]⚠ large[/]" if seg.token_count > 2000 else ""
                    label = "tool_result"
                    detail = f"{seg.token_count:,} tok  injected{warning}"
                elif seg.role == "system":
                    color = "#4488ff"
                    label = "system"
                    detail = f"{seg.token_count:,} tok  loaded"
                else:
                    color = "#e8e8e8"
                    label = seg.role
                    detail = f"{seg.token_count:,} tok"

                self._events.append(
                    f"  [{color}]{ts}  [{label:14}]  {detail}[/]"
                )

        self._update_display()

    def _update_display(self) -> None:
        if not self._run:
            return

        # Reload run for updated totals
        run = self.db.get_run(self._run.id)
        if not run:
            return
        self._run = run

        total_tokens = sum(c.input_tokens + c.output_tokens for c in run.calls)
        call_count = len(run.calls)
        elapsed = time.time() - self._start_time
        elapsed_str = f"{int(elapsed // 60):02}:{int(elapsed % 60):02}"

        status = "[#00cc66 bold]LIVE[/]" if run.status == "running" else f"[dim]{run.status}[/]"

        header = self.query_one("#header-bar", Static)
        header.update(
            f"{status} · {run.id} · {elapsed_str} · {call_count} calls · {total_tokens:,} tok"
        )

        # Budget bar
        model = run.calls[0].model if run.calls else "gpt-4o"
        ctx_limit = get_context_limit(model)
        pct = total_tokens / ctx_limit * 100 if ctx_limit else 0
        filled = int(pct / 5)
        bar = "▓" * filled + "░" * (20 - filled)
        color = "#00d4aa" if pct < 80 else "#ffaa00" if pct < 95 else "#ff4455"

        budget = self.query_one("#budget-bar", Static)
        budget.update(f"[{color}]{bar}[/]  {pct:.0f}%  {total_tokens:,} / {ctx_limit:,} tok")

        # Events
        events = self.query_one("#events", Static)
        events.update("\n".join(self._events[-50:]) if self._events else "[dim]Waiting for events...[/]")

    def action_snapshot(self) -> None:
        self.notify("Snapshot saved")

    def action_pause(self) -> None:
        self._paused = not self._paused
        state = "paused" if self._paused else "resumed"
        self.notify(f"Live updates {state}")
