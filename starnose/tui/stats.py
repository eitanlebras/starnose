"""snose stats — TUI for aggregate run statistics."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static

from starnose.db import StatsResult


CSS = """
Screen {
    background: #0d0d0d;
}

#summary {
    height: auto;
    border: solid #2a2a2a;
    padding: 1;
    margin-bottom: 1;
}

#breakdown {
    height: auto;
    border: solid #2a2a2a;
    padding: 1;
    margin-bottom: 1;
}

#insights {
    height: auto;
    border: solid #2a2a2a;
    padding: 1;
    margin-bottom: 1;
}

#run-table {
    height: 1fr;
    border: solid #2a2a2a;
    padding: 1;
}
"""


class StatsApp(App):
    """TUI for aggregate stats."""

    TITLE = "snose stats"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, stats: StatsResult):
        super().__init__()
        self.stats = stats

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(self._build_summary(), id="summary")
            yield Static(self._build_breakdown(), id="breakdown")
            yield Static(self._build_insights(), id="insights")
            yield Static(self._build_run_table(), id="run-table")
        yield Footer()

    def _build_summary(self) -> str:
        s = self.stats
        success_rate = s.success_count / s.total_runs * 100 if s.total_runs else 0
        avg_latency_s = s.avg_latency_ms / 1000

        # Token efficiency: output / input ratio
        efficiency = (
            s.avg_output_tokens / s.avg_input_tokens
            if s.avg_input_tokens > 0
            else 0
        )

        return (
            "[bold]Summary[/]\n\n"
            f"  Avg context usage: [bold]{s.avg_tokens:,.0f}[/] tok    "
            f"Success rate: [bold]{success_rate:.0f}%[/]     "
            f"Avg latency: [bold]{avg_latency_s:.1f}s[/]\n"
            f"  Avg tokens/run: [bold]{s.avg_tokens:,.0f}[/]    "
            f"Total runs: [bold]{s.total_runs}[/]          "
            f"Token efficiency: [bold]{efficiency:.2f}[/]"
        )

    def _build_breakdown(self) -> str:
        seg = self.stats.segment_breakdown
        if not seg:
            return "[bold]Segment Breakdown[/]\n\n  [dim]No data[/]"

        total = sum(seg.values())
        lines = ["[bold]Segment Breakdown[/]\n"]

        for seg_type, count in sorted(seg.items(), key=lambda x: -x[1]):
            share = count / total * 100 if total else 0
            bar_len = int(share / 5)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {seg_type:16}  {share:>4.0f}%  [#00d4aa]{bar}[/]")

        return "\n".join(lines)

    def _build_insights(self) -> str:
        s = self.stats
        lines = ["[bold]Insights[/]\n"]

        # Budget vs failure correlation
        high_budget_fails = 0
        high_budget_total = 0
        for run in s.runs:
            tok = sum(c.input_tokens + c.output_tokens for c in run.calls)
            model = run.calls[0].model if run.calls else "gpt-4o"
            from starnose.tokens import get_context_limit
            limit = get_context_limit(model)
            if tok / limit > 0.8:
                high_budget_total += 1
                if run.status == "failed":
                    high_budget_fails += 1

        if high_budget_total > 0:
            fail_pct = high_budget_fails / high_budget_total * 100
            lines.append(
                f"  ► Runs exceeding 80% context budget fail {fail_pct:.0f}% of the time"
            )

        # Tool result growth trend
        if len(s.runs) >= 3:
            tool_totals = []
            for run in reversed(s.runs):
                tt = sum(
                    seg.token_count
                    for c in run.calls
                    for seg in c.segments
                    if seg.seg_type == "tool_result"
                )
                tool_totals.append(tt)

            if len(tool_totals) >= 2 and tool_totals[0] > 0:
                growth = (tool_totals[-1] - tool_totals[0]) / tool_totals[0] * 100
                if abs(growth) > 10:
                    lines.append(
                        f"  ► Tool results have {'grown' if growth > 0 else 'shrunk'} "
                        f"{abs(growth):.0f}% over your last {len(s.runs)} runs"
                    )

        # Latency correlation
        if len(s.runs) >= 3:
            tokens = []
            latencies = []
            for run in s.runs:
                t = sum(c.input_tokens + c.output_tokens for c in run.calls)
                l = sum(c.latency_ms for c in run.calls)
                if t > 0 and l > 0:
                    tokens.append(t)
                    latencies.append(l)

            if len(tokens) >= 3:
                # Simple correlation check
                mean_t = sum(tokens) / len(tokens)
                mean_l = sum(latencies) / len(latencies)
                cov = sum((t - mean_t) * (l - mean_l) for t, l in zip(tokens, latencies))
                var_t = sum((t - mean_t) ** 2 for t in tokens)
                var_l = sum((l - mean_l) ** 2 for l in latencies)
                if var_t > 0 and var_l > 0:
                    r = cov / (var_t * var_l) ** 0.5
                    if r > 0.7:
                        lines.append(
                            f"  ► Latency strongly correlates with token count (r={r:.2f})"
                        )

        if len(lines) == 1:
            lines.append("  [dim]Not enough data for insights yet.[/]")

        return "\n".join(lines)

    def _build_run_table(self) -> str:
        lines = ["[bold]Recent Runs[/]\n"]
        lines.append(
            f"  {'ID':12}  {'NAME':20}  {'TOKENS':>8}  {'STATUS':>8}  {'LATENCY':>10}  {'AGE':>8}"
        )
        lines.append(f"  {'─' * 75}")

        from starnose.cli import _format_age

        for run in self.stats.runs[:10]:
            tok = sum(c.input_tokens + c.output_tokens for c in run.calls)
            lat = sum(c.latency_ms for c in run.calls)
            icon = {"success": "[green]✓[/]", "failed": "[red]✗[/]"}.get(
                run.status, "[yellow]●[/]"
            )
            lines.append(
                f"  {run.id:12}  {(run.name or '—'):20}  {tok:>8,}  {icon:>8}  "
                f"{lat:>8,}ms  {_format_age(run.started_at):>8}"
            )

        return "\n".join(lines)
