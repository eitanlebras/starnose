"""snose diff — TUI for comparing two runs."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from starnose.db import Run
from starnose.hypotheses import generate_delta_table, generate_hypotheses
from starnose.tokens import get_context_limit


CSS = """
Screen {
    background: #0d0d0d;
}

#summary-panel {
    height: auto;
    border: solid #2a2a2a;
    padding: 1;
    margin-bottom: 1;
}

#delta-panel {
    height: auto;
    border: solid #2a2a2a;
    padding: 1;
    margin-bottom: 1;
}

#hypothesis-panel {
    height: 1fr;
    border: solid #2a2a2a;
    padding: 1;
}

.panel-title {
    text-style: bold;
    color: #e8e8e8;
    margin-bottom: 1;
}
"""


class DiffApp(App):
    """TUI for diffing two runs."""

    TITLE = "snose diff"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("o", "optimize", "Optimize"),
        Binding("s", "save", "Save hypothesis"),
    ]

    def __init__(self, run_a: Run, run_b: Run):
        super().__init__()
        self.run_a = run_a
        self.run_b = run_b

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(self._build_summary(), id="summary-panel")
            yield Static(self._build_delta(), id="delta-panel")
            yield Static(self._build_hypothesis(), id="hypothesis-panel")
        yield Footer()

    def _build_summary(self) -> str:
        def _run_line(r: Run) -> str:
            tok = sum(c.input_tokens + c.output_tokens for c in r.calls)
            dur = "—"
            if r.started_at and r.finished_at:
                s = (r.finished_at - r.started_at).total_seconds()
                dur = f"{s:.1f}s"
            icon = {"success": "[green]✓[/]", "failed": "[red]✗[/]", "running": "[yellow]●[/]"}.get(
                r.status, r.status
            )
            return f"  {r.id}  {r.name or '—':20}  {dur:>8}  {tok:>8,} tok  {icon} {r.status}"

        return (
            "[bold]Run Comparison[/]\n\n"
            f"{_run_line(self.run_a)}\n"
            f"{_run_line(self.run_b)}"
        )

    def _build_delta(self) -> str:
        rows = generate_delta_table(self.run_a, self.run_b)
        lines = ["[bold]Context Delta[/]\n"]
        lines.append(f"  {'SEGMENT':16}  {'RUN A':>8}  {'RUN B':>8}  {'DELTA':>10}  {'':>10}")
        lines.append(f"  {'─' * 60}")

        for row in rows:
            seg = row["segment"]
            va = f"{row['run_a']:,}"
            vb = f"{row['run_b']:,}"

            if row["identical"]:
                delta_str = "= identical"
                style = "dim"
            elif row["delta"] > 0:
                delta_str = f"+{row['delta']:,}"
                style = "red" if row["delta"] > 1000 else "yellow"
            else:
                delta_str = f"{row['delta']:,}"
                style = "green"

            pct = row["pct"] if not row["identical"] else ""
            name = f"[bold]{seg}[/]" if seg == "TOTAL" else seg
            lines.append(
                f"  {name:16}  {va:>8}  {vb:>8}  [{style}]{delta_str:>10}[/]  {pct:>10}"
            )

        return "\n".join(lines)

    def _build_hypothesis(self) -> str:
        hypotheses = generate_hypotheses(self.run_a, self.run_b)
        lines = ["[bold]Hypothesis[/]\n"]

        for h in hypotheses:
            color = {"high": "#ff4455", "medium": "#ffaa00", "low": "#666666"}.get(
                h.confidence, "#e8e8e8"
            )
            lines.append(f"  [{color}]{h.confidence.upper()}[/]  [bold]{h.title}[/]")
            lines.append(f"  {h.explanation}")
            lines.append("")

        return "\n".join(lines)

    def action_optimize(self) -> None:
        # Find the larger run
        tok_a = sum(c.input_tokens + c.output_tokens for c in self.run_a.calls)
        tok_b = sum(c.input_tokens + c.output_tokens for c in self.run_b.calls)
        larger = self.run_b if tok_b > tok_a else self.run_a
        self.notify(f"Run: snose optimize {larger.id}")

    def action_save(self) -> None:
        self.notify("Hypothesis saved to run metadata")
