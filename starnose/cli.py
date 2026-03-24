"""starnose CLI — context window observability for LLM agents."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from starnose.db import Database
from starnose.proxy import create_proxy_app, find_free_port, run_proxy_server
from starnose.tokens import get_context_limit
from starnose.certs import CA_CERT_PATH, get_or_create_ca

app = typer.Typer(
    name="snose",
    help="Context window observability for LLM agents.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

_CONFIG_PATH = Path("~/.starnose/config.json").expanduser()


def _get_db(db_path: str | None = None) -> Database:
    return Database(db_path)


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _parse_since(since: str | None) -> datetime | None:
    if not since:
        return None
    now = datetime.now(timezone.utc)
    amount = int(since[:-1])
    unit = since[-1]
    if unit == "h":
        return now - timedelta(hours=amount)
    elif unit == "d":
        return now - timedelta(days=amount)
    elif unit == "m":
        return now - timedelta(minutes=amount)
    return None


def _format_age(dt: datetime | None) -> str:
    if not dt:
        return "—"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _format_duration(start: datetime | None, end: datetime | None) -> str:
    if not start:
        return "—"
    if not end:
        end = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - start
    secs = delta.total_seconds()
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}m"


def _status_icon(status: str) -> str:
    return {"success": "[green]✓[/]", "failed": "[red]✗[/]", "running": "[yellow]●[/]"}.get(
        status, status
    )


# ── snose setup ──────────────────────────────────────────────────────────────


@app.command()
def setup():
    """One-time setup: generate CA cert for MITM proxy (needed for Claude Code)."""
    console.print("[bold]starnose setup[/]\n")

    ca_cert, _ = get_or_create_ca()
    console.print(f"  CA cert: [green]{CA_CERT_PATH}[/]")

    # Check if CA is already trusted in macOS keychain
    import subprocess as sp
    try:
        result = sp.run(
            ["security", "verify-cert", "-c", str(CA_CERT_PATH)],
            capture_output=True, text=True,
        )
        already_trusted = result.returncode == 0
    except FileNotFoundError:
        already_trusted = False

    if already_trusted:
        console.print("  Status: [green]CA already trusted in system keychain[/]")
        console.print("\n[green]Ready![/] Run: [bold]snose run -- claude[/]")
        return

    console.print("  Status: [yellow]CA not yet trusted[/]\n")
    console.print("  To trust the CA (required for intercepting Claude Code):\n")
    cmd = f"sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain {CA_CERT_PATH}"
    console.print(f"    {cmd}", highlight=False)
    console.print()

    if typer.confirm("Run this command now? (requires sudo)"):
        ret = sp.call([
            "sudo", "security", "add-trusted-cert",
            "-d", "-r", "trustRoot",
            "-k", "/Library/Keychains/System.keychain",
            str(CA_CERT_PATH),
        ])
        if ret == 0:
            console.print("\n[green]CA trusted![/] Run: [bold]snose run -- claude[/]")
        else:
            console.print("\n[red]Failed to trust CA.[/] Run the command manually above.")
    else:
        console.print("[dim]Skipped. You can run the command above manually.[/]")


# ── snose run ────────────────────────────────────────────────────────────────


def _needs_mitm(command: list[str]) -> bool:
    """Detect if the command needs MITM proxy (vs OpenAI-compatible proxy)."""
    if not command:
        return False
    binary = os.path.basename(command[0])
    # Claude Code and other agents that use their own auth
    return binary in ("claude", "claude-code")


@app.command()
def run(
    command: list[str] = typer.Argument(help="Command to run"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for this run"),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Tags (repeatable)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model label"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Auto-diff against last run"),
    no_proxy: bool = typer.Option(False, "--no-proxy", help="Disable proxy, just exec"),
    mitm: bool = typer.Option(False, "--mitm", help="Use MITM proxy (auto-detected for claude)"),
    db: Optional[str] = typer.Option(None, "--db", help="Database path"),
):
    """Run a command with context window interception."""
    if not command:
        console.print("[red]No command specified.[/]")
        raise typer.Exit(1)

    database = _get_db(db)
    tags = tag or []
    run_record = database.create_run(name=name, tags=tags)
    run_id = run_record.id

    # Auto-detect MITM mode for Claude Code
    use_mitm = mitm or _needs_mitm(command)

    console.print(f"[dim]starnose[/] · run [bold]{run_id}[/]", highlight=False)
    if name:
        console.print(f"  name: {name}")
    if tags:
        console.print(f"  tags: {', '.join(tags)}")

    env = os.environ.copy()
    env["STARNOSE_RUN_ID"] = run_id
    if name:
        env["STARNOSE_RUN_NAME"] = name

    if no_proxy:
        console.print("  proxy: [dim]disabled[/]")
        console.print()
        returncode = subprocess.call(command, env=env)

    elif use_mitm:
        # MITM proxy mode — intercepts HTTPS via CONNECT tunnel
        from starnose.mitm import find_free_port as mitm_free_port, run_mitm_server

        # Ensure CA exists
        if not CA_CERT_PATH.exists():
            console.print("[yellow]No CA cert found. Running setup...[/]\n")
            get_or_create_ca()
            console.print(f"  CA cert: {CA_CERT_PATH}")
            console.print("  [yellow]You may need to trust this CA. Run: snose setup[/]\n")

        port = mitm_free_port()
        console.print(f"  mode: [bold]MITM[/]")
        console.print(f"  proxy: [bold]http://127.0.0.1:{port}[/]")
        console.print(f"  CA: {CA_CERT_PATH}")
        console.print()

        # Route HTTPS traffic through our MITM proxy
        env["HTTPS_PROXY"] = f"http://127.0.0.1:{port}"
        env["HTTP_PROXY"] = f"http://127.0.0.1:{port}"
        env["ALL_PROXY"] = f"http://127.0.0.1:{port}"
        # Tell Node.js to trust our CA (for Claude Code and other Node agents)
        env["NODE_EXTRA_CA_CERTS"] = str(CA_CERT_PATH)
        # Tell Python requests/httpx to trust our CA
        env["REQUESTS_CA_BUNDLE"] = str(CA_CERT_PATH)
        env["SSL_CERT_FILE"] = str(CA_CERT_PATH)

        proxy_thread = threading.Thread(
            target=run_mitm_server,
            args=(database, run_id, port),
            daemon=True,
        )
        proxy_thread.start()
        time.sleep(0.3)

        returncode = subprocess.call(command, env=env)

    else:
        # Standard OpenAI-compatible proxy mode
        port = find_free_port()
        console.print(f"  mode: [bold]proxy[/]")
        console.print(f"  proxy: [bold]http://127.0.0.1:{port}[/]")
        console.print()

        # Save original upstream URLs before overwriting
        original_openai_base = env.get("OPENAI_BASE_URL", "https://api.openai.com")
        original_anthropic_base = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

        # Set upstream env vars so proxy knows where to forward
        env["STARNOSE_OPENAI_UPSTREAM"] = original_openai_base.rstrip("/")
        env["STARNOSE_ANTHROPIC_UPSTREAM"] = original_anthropic_base.rstrip("/")

        # Point clients at our proxy
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

        proxy_thread = threading.Thread(
            target=run_proxy_server,
            args=(database, run_id, port),
            daemon=True,
        )
        proxy_thread.start()
        time.sleep(0.3)

        returncode = subprocess.call(command, env=env)

    # Brief pause to let proxy threads flush any pending recordings
    time.sleep(0.5)

    # Finalize
    status = "success" if returncode == 0 else "failed"
    database.update_run_status(run_id, status)

    # Reload run to get final stats
    final_run = database.get_run(run_id)
    total_tokens = sum(c.input_tokens + c.output_tokens for c in final_run.calls) if final_run else 0
    call_count = len(final_run.calls) if final_run else 0
    duration = _format_duration(
        final_run.started_at if final_run else None,
        final_run.finished_at if final_run else None,
    )

    console.print()
    console.print(
        f"[dim]starnose[/] · {_status_icon(status)} {status} · "
        f"{call_count} calls · {total_tokens:,} tokens · {duration}"
    )

    if compare and final_run:
        runs = database.get_last_n_runs(2)
        if len(runs) >= 2:
            console.print()
            _print_diff(runs[0], runs[1])

    raise typer.Exit(returncode)


# ── snose ls ─────────────────────────────────────────────────────────────────


@app.command(name="ls")
def ls_runs(
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t"),
    last: int = typer.Option(20, "--last", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """List recorded runs."""
    database = _get_db(db)
    runs = database.list_runs(limit=last, tags=tag)

    if as_json:
        data = [
            {
                "id": r.id,
                "name": r.name,
                "tags": r.tags,
                "status": r.status,
                "tokens": sum(c.input_tokens + c.output_tokens for c in r.calls),
                "calls": len(r.calls),
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in runs
        ]
        console.print_json(json.dumps(data))
        return

    if not runs:
        console.print("[dim]No runs recorded yet. Try: snose run python my_agent.py[/]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Tags", style="dim")
    table.add_column("Tokens", justify="right")
    table.add_column("Status")
    table.add_column("Calls", justify="right")
    table.add_column("Age", style="dim")

    for r in runs:
        tokens = sum(c.input_tokens + c.output_tokens for c in r.calls)
        table.add_row(
            r.id,
            r.name or "—",
            ", ".join(r.tags) if r.tags else "—",
            f"{tokens:,}",
            _status_icon(r.status),
            str(len(r.calls)),
            _format_age(r.started_at),
        )

    console.print(table)


# ── snose export ─────────────────────────────────────────────────────────────


@app.command()
def export(
    run_id: Optional[str] = typer.Argument(None, help="Run ID to export"),
    format: str = typer.Option("json", "--format", "-f", help="json or jsonl"),
    last: Optional[int] = typer.Option(None, "--last", "-n"),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Export run data to JSON/JSONL."""
    database = _get_db(db)

    if run_id:
        runs_to_export = [database.get_run(run_id)]
        if runs_to_export[0] is None:
            console.print(f"[red]Run {run_id} not found.[/]")
            raise typer.Exit(1)
    else:
        limit = last or 1
        runs_to_export = database.list_runs(limit=limit, tags=tag)

    for r in runs_to_export:
        data = {
            "run": {
                "id": r.id,
                "name": r.name,
                "tags": r.tags,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "metadata": r.metadata,
            },
            "calls": [
                {
                    "id": c.id,
                    "sequence": c.sequence,
                    "model": c.model,
                    "params": c.params,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "latency_ms": c.latency_ms,
                    "finish_reason": c.finish_reason,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "segments": [
                        {
                            "role": s.role,
                            "seg_type": s.seg_type,
                            "content": s.content,
                            "token_count": s.token_count,
                            "position": s.position,
                        }
                        for s in c.segments
                    ],
                    "messages": [
                        {"role": s.role, "content": s.content}
                        for s in c.segments
                    ],
                }
                for c in r.calls
            ],
        }
        if format == "jsonl":
            print(json.dumps(data))
        else:
            print(json.dumps(data, indent=2))


# ── snose inspect ────────────────────────────────────────────────────────────


@app.command()
def inspect(
    run_id: Optional[str] = typer.Argument(None, help="Run ID"),
    last: bool = typer.Option(False, "--last", "-l", help="Pick from recent runs"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Inspect a run's context window (TUI)."""
    database = _get_db(db)

    if last:
        runs = database.get_last_n_runs(10)
        if not runs:
            console.print("[dim]No runs found.[/]")
            raise typer.Exit(1)
        # Show picker and launch TUI
        _print_run_picker(runs)
        choice = typer.prompt("Select run number", type=int, default=1)
        if 1 <= choice <= len(runs):
            target_run = runs[choice - 1]
        else:
            console.print("[red]Invalid selection.[/]")
            raise typer.Exit(1)
    elif run_id:
        target_run = database.get_run(run_id)
        if not target_run:
            console.print(f"[red]Run {run_id} not found.[/]")
            raise typer.Exit(1)
    else:
        target_run = database.get_last_run()
        if not target_run:
            console.print("[dim]No runs found. Try: snose run python my_agent.py[/]")
            raise typer.Exit(1)

    try:
        from starnose.tui.inspect import InspectApp
        tui = InspectApp(target_run, database)
        tui.run()
    except Exception:
        _print_inspect_fallback(target_run)


# ── snose diff ───────────────────────────────────────────────────────────────


@app.command()
def diff(
    id_a: Optional[str] = typer.Argument(None, help="First run ID"),
    id_b: Optional[str] = typer.Argument(None, help="Second run ID"),
    last: bool = typer.Option(False, "--last", "-l", help="Pick from recent runs"),
    as_json: bool = typer.Option(False, "--json"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Diff two runs' context windows."""
    database = _get_db(db)

    if last:
        runs = database.get_last_n_runs(10)
        if len(runs) < 2:
            console.print("[red]Need at least 2 runs to diff.[/]")
            raise typer.Exit(1)
        _print_run_picker(runs)
        a = typer.prompt("Select run A", type=int, default=1)
        b = typer.prompt("Select run B", type=int, default=2)
        run_a = runs[a - 1]
        run_b = runs[b - 1]
    elif id_a and id_b:
        run_a = database.get_run(id_a)
        run_b = database.get_run(id_b)
        if not run_a or not run_b:
            console.print("[red]One or both runs not found.[/]")
            raise typer.Exit(1)
    else:
        runs = database.get_last_n_runs(2)
        if len(runs) < 2:
            console.print("[red]Need at least 2 runs to diff.[/]")
            raise typer.Exit(1)
        run_a, run_b = runs[1], runs[0]  # older first

    if as_json:
        from starnose.hypotheses import generate_hypotheses, generate_delta_table
        hyps = generate_hypotheses(run_a, run_b)
        delta = generate_delta_table(run_a, run_b)
        data = {
            "run_a": run_a.id,
            "run_b": run_b.id,
            "delta": delta,
            "hypotheses": [
                {"title": h.title, "explanation": h.explanation, "confidence": h.confidence}
                for h in hyps
            ],
        }
        console.print_json(json.dumps(data))
        return

    try:
        from starnose.tui.diff import DiffApp
        tui = DiffApp(run_a, run_b)
        tui.run()
    except Exception:
        _print_diff(run_a, run_b)


# ── snose watch ──────────────────────────────────────────────────────────────


@app.command()
def watch(
    pid: Optional[int] = typer.Option(None, "--pid", "-p", help="PID to watch"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Watch a live run's context window."""
    database = _get_db(db)
    try:
        from starnose.tui.watch import WatchApp
        tui = WatchApp(database)
        tui.run()
    except Exception:
        console.print("[red]Failed to launch watch TUI.[/]")
        raise typer.Exit(1)


# ── snose stats ──────────────────────────────────────────────────────────────


@app.command()
def stats(
    last: int = typer.Option(20, "--last", "-n"),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t"),
    since: Optional[str] = typer.Option(None, "--since", help="e.g. 1h, 24h, 7d"),
    as_json: bool = typer.Option(False, "--json"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Show aggregate stats across runs."""
    database = _get_db(db)
    since_dt = _parse_since(since)
    runs = database.list_runs(limit=last, tags=tag, since=since_dt)

    if not runs:
        console.print("[dim]No runs found.[/]")
        return

    run_ids = [r.id for r in runs]
    result = database.get_stats(run_ids)

    if as_json:
        data = {
            "total_runs": result.total_runs,
            "success_count": result.success_count,
            "failed_count": result.failed_count,
            "avg_tokens": result.avg_tokens,
            "avg_latency_ms": result.avg_latency_ms,
            "segment_breakdown": result.segment_breakdown,
        }
        console.print_json(json.dumps(data))
        return

    try:
        from starnose.tui.stats import StatsApp
        tui = StatsApp(result)
        tui.run()
    except Exception:
        _print_stats_fallback(result)


# ── snose optimize ───────────────────────────────────────────────────────────


@app.command()
def optimize(
    run_id: Optional[str] = typer.Argument(None),
    last: int = typer.Option(1, "--last", "-n"),
    apply: Optional[int] = typer.Option(None, "--apply", help="Apply suggestion N"),
    db: Optional[str] = typer.Option(None, "--db"),
):
    """Analyze runs and suggest optimizations."""
    database = _get_db(db)

    if run_id:
        target = database.get_run(run_id)
        if not target:
            console.print(f"[red]Run {run_id} not found.[/]")
            raise typer.Exit(1)
        runs = [target]
    else:
        runs = database.list_runs(limit=last)

    if not runs:
        console.print("[dim]No runs found.[/]")
        return

    suggestions = _generate_suggestions(runs)

    if apply is not None:
        if 1 <= apply <= len(suggestions):
            s = suggestions[apply - 1]
            console.print(f"\n[bold]Applying suggestion #{apply}: {s['title']}[/]\n")
            console.print(s.get("code_hint", "[dim]No code change available for this suggestion.[/]"))
        else:
            console.print(f"[red]Invalid suggestion index. Valid: 1-{len(suggestions)}[/]")
        return

    if not suggestions:
        console.print("[green]No optimization suggestions — context usage looks efficient.[/]")
        return

    console.print(f"\n[bold]Optimization Suggestions[/] ({len(suggestions)} found)\n")

    for i, s in enumerate(suggestions, 1):
        impact_color = {"HIGH": "red", "MED": "yellow", "LOW": "dim"}.get(s["impact"], "white")
        console.print(
            f"  [{impact_color}]{s['impact']}[/]  "
            f"[bold]#{i}. {s['title']}[/]"
        )
        console.print(f"       {s['description']}")
        console.print(f"       [green]Estimated savings: ~{s['savings']:,} tokens[/]")
        console.print(f"       [dim]Apply: snose optimize --apply {i}[/]")
        console.print()


def _generate_suggestions(runs: list) -> list[dict]:
    """Generate optimization suggestions from runs."""
    suggestions = []

    for run in runs:
        for call in run.calls:
            for seg in call.segments:
                # 1. Tool result bloat
                if seg.seg_type == "tool_result" and seg.token_count > 3000:
                    savings = (seg.token_count - 1500) * max(1, len(run.calls))
                    suggestions.append({
                        "title": "Large tool result detected",
                        "description": (
                            f"Tool result in call #{call.sequence} returned "
                            f"{seg.token_count:,} tokens. Consider truncating to top-k chunks."
                        ),
                        "impact": "HIGH",
                        "savings": savings,
                        "code_hint": (
                            "Before:\n"
                            f"  # Tool result: {seg.content[:80]}...\n"
                            f"  # {seg.token_count:,} tokens\n\n"
                            "After:\n"
                            "  # Truncate tool output to relevant chunks:\n"
                            "  result = tool_output[:1500]  # or use semantic chunking\n"
                            "  # ~1,500 tokens"
                        ),
                    })

    # 2. Check for repeated content across calls
    if runs:
        content_freq: dict[str, int] = {}
        for run in runs:
            for call in run.calls:
                for seg in call.segments:
                    if seg.token_count > 50:
                        key = seg.content[:200]
                        content_freq[key] = content_freq.get(key, 0) + 1

        for snippet, count in content_freq.items():
            if count > 3:
                suggestions.append({
                    "title": "Repeated context block",
                    "description": (
                        f"'{snippet[:60]}...' appears in {count} consecutive calls. "
                        "Move to system prompt or deduplicate."
                    ),
                    "impact": "MED",
                    "savings": count * 100,
                    "code_hint": (
                        "Move repeated content to the system prompt or cache it:\n\n"
                        f"  # This block appears {count} times:\n"
                        f"  # \"{snippet[:80]}...\"\n"
                        "  # -> Move to system prompt for single injection"
                    ),
                })

    # 3. System prompt redundancy (simple token overlap check)
    for run in runs:
        sys_segments = []
        for call in run.calls:
            for seg in call.segments:
                if seg.seg_type == "system_prompt" and seg.content:
                    sys_segments.append(seg)

        if sys_segments:
            for seg in sys_segments:
                lines = seg.content.split("\n")
                seen_lines: dict[str, list[int]] = {}
                for li, line in enumerate(lines):
                    stripped = line.strip()
                    if len(stripped) > 30:
                        if stripped in seen_lines:
                            seen_lines[stripped].append(li)
                        else:
                            seen_lines[stripped] = [li]

                for line_text, positions in seen_lines.items():
                    if len(positions) > 1:
                        suggestions.append({
                            "title": "System prompt redundancy",
                            "description": (
                                f"Lines {positions} contain duplicate content. "
                                f"Estimated saving: ~{len(positions) * 10} tokens."
                            ),
                            "impact": "LOW",
                            "savings": len(positions) * 10,
                            "code_hint": (
                                f"Remove duplicate line:\n"
                                f"  \"{line_text[:80]}...\""
                            ),
                        })
                        break  # One per system prompt

    # Sort by impact
    priority = {"HIGH": 0, "MED": 1, "LOW": 2}
    suggestions.sort(key=lambda s: priority.get(s["impact"], 3))

    return suggestions


# ── snose config ─────────────────────────────────────────────────────────────


@app.command()
def config(
    action: Optional[str] = typer.Argument(None, help="set or reset"),
    key: Optional[str] = typer.Argument(None),
    value: Optional[str] = typer.Argument(None),
):
    """View or modify starnose configuration."""
    cfg = _load_config()

    if action is None:
        # Show current config
        if not cfg:
            console.print("[dim]No custom configuration. Using defaults.[/]")
            console.print("[dim]  db: ~/.starnose/runs.db[/]")
            console.print("[dim]  model: gpt-4o[/]")
            console.print("[dim]  context_limit: 128000[/]")
            return

        table = Table(box=box.SIMPLE)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)

    elif action == "set":
        if not key or value is None:
            console.print("[red]Usage: snose config set <key> <value>[/]")
            raise typer.Exit(1)
        # Try to parse as number
        try:
            cfg[key] = int(value)
        except ValueError:
            cfg[key] = value
        _save_config(cfg)
        console.print(f"[green]Set {key} = {value}[/]")

    elif action == "reset":
        _save_config({})
        console.print("[green]Config reset to defaults.[/]")

    else:
        console.print(f"[red]Unknown action: {action}. Use 'set' or 'reset'.[/]")


# ── Fallback/rich output helpers ─────────────────────────────────────────────


def _print_run_picker(runs: list) -> None:
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="dim")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Tokens", justify="right")
    table.add_column("Status")
    table.add_column("Age", style="dim")

    for i, r in enumerate(runs, 1):
        tokens = sum(c.input_tokens + c.output_tokens for c in r.calls)
        table.add_row(
            str(i),
            r.id,
            r.name or "—",
            f"{tokens:,}",
            _status_icon(r.status),
            _format_age(r.started_at),
        )
    console.print(table)


def _print_inspect_fallback(run) -> None:
    """Rich table fallback when TUI fails."""
    total_tokens = sum(c.input_tokens + c.output_tokens for c in run.calls)
    model = run.calls[0].model if run.calls else "unknown"
    ctx_limit = get_context_limit(model)
    pct = total_tokens / ctx_limit * 100 if ctx_limit else 0

    console.print(f"\n[bold]Run {run.id}[/]  {run.name or ''}")
    console.print(f"  Status: {_status_icon(run.status)}  Calls: {len(run.calls)}  Tokens: {total_tokens:,}")
    console.print(f"  Budget: {pct:.0f}%  ({total_tokens:,} / {ctx_limit:,})")
    console.print(f"  Duration: {_format_duration(run.started_at, run.finished_at)}")

    # Segment breakdown
    seg_totals: dict[str, int] = {}
    for call in run.calls:
        for seg in call.segments:
            seg_totals[seg.seg_type] = seg_totals.get(seg.seg_type, 0) + seg.token_count

    if seg_totals:
        console.print("\n[bold]Segments[/]")
        table = Table(box=box.SIMPLE)
        table.add_column("Type")
        table.add_column("Tokens", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Bar")

        input_total = sum(seg_totals.values())
        for seg_type, count in sorted(seg_totals.items(), key=lambda x: -x[1]):
            share = count / input_total * 100 if input_total else 0
            bar_len = int(share / 5)
            bar = "[bold green]" + "▓" * bar_len + "[/]" + "░" * (20 - bar_len)
            table.add_row(seg_type, f"{count:,}", f"{share:.0f}%", bar)
        console.print(table)

    # Call list
    if run.calls:
        console.print("\n[bold]Calls[/]")
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="dim")
        table.add_column("Model")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Finish")

        for c in run.calls:
            table.add_row(
                str(c.sequence),
                c.model or "—",
                f"{c.input_tokens:,}",
                f"{c.output_tokens:,}",
                f"{c.latency_ms:,}ms",
                c.finish_reason or "—",
            )
        console.print(table)


def _print_diff(run_a, run_b) -> None:
    """Rich table diff fallback."""
    from starnose.hypotheses import generate_hypotheses, generate_delta_table

    tok_a = sum(c.input_tokens + c.output_tokens for c in run_a.calls)
    tok_b = sum(c.input_tokens + c.output_tokens for c in run_b.calls)

    console.print("[bold]Run Comparison[/]\n")
    console.print(
        f"  {run_a.id}  {run_a.name or '—':20}  "
        f"{_format_duration(run_a.started_at, run_a.finished_at):>8}  "
        f"{tok_a:>8,} tok  {_status_icon(run_a.status)}"
    )
    console.print(
        f"  {run_b.id}  {run_b.name or '—':20}  "
        f"{_format_duration(run_b.started_at, run_b.finished_at):>8}  "
        f"{tok_b:>8,} tok  {_status_icon(run_b.status)}"
    )

    delta_rows = generate_delta_table(run_a, run_b)
    console.print()

    table = Table(box=box.SIMPLE, title="Context Delta")
    table.add_column("Segment")
    table.add_column("Run A", justify="right")
    table.add_column("Run B", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("", style="dim")

    for row in delta_rows:
        delta_str = f"{row['delta']:+,}" if row["delta"] != 0 else "="
        style = ""
        if row["delta"] > 0:
            style = "red" if row["delta"] > 1000 else "yellow"
        elif row["delta"] < 0:
            style = "green"

        table.add_row(
            f"[bold]{row['segment']}[/]" if row["segment"] == "TOTAL" else row["segment"],
            f"{row['run_a']:,}",
            f"{row['run_b']:,}",
            f"[{style}]{delta_str}[/]" if style else delta_str,
            row["pct"] if not row["identical"] else "identical",
        )
    console.print(table)

    # Hypotheses
    hypotheses = generate_hypotheses(run_a, run_b)
    console.print("\n[bold]Hypothesis[/]\n")
    for h in hypotheses:
        conf_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(h.confidence, "white")
        console.print(f"  [{conf_color}]{h.confidence.upper()}[/]  [bold]{h.title}[/]")
        console.print(f"  {h.explanation}")
        console.print()


def _print_stats_fallback(result) -> None:
    """Rich table stats fallback."""
    success_rate = (
        result.success_count / result.total_runs * 100 if result.total_runs else 0
    )

    console.print("\n[bold]Aggregate Stats[/]\n")
    console.print(f"  Runs: {result.total_runs}   Success rate: {success_rate:.0f}%")
    console.print(f"  Avg tokens/run: {result.avg_tokens:,.0f}   Avg latency: {result.avg_latency_ms:,.0f}ms")

    if result.segment_breakdown:
        total_segs = sum(result.segment_breakdown.values())
        console.print("\n[bold]Segment Breakdown[/]")

        table = Table(box=box.SIMPLE)
        table.add_column("Type")
        table.add_column("Tokens", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Bar")

        for seg_type, count in sorted(
            result.segment_breakdown.items(), key=lambda x: -x[1]
        ):
            share = count / total_segs * 100 if total_segs else 0
            bar_len = int(share / 5)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            table.add_row(seg_type, f"{count:,}", f"{share:.0f}%", bar)
        console.print(table)

    # Insights
    console.print("\n[bold]Insights[/]\n")
    if result.failed_count > 0 and result.total_runs > 2:
        fail_rate = result.failed_count / result.total_runs * 100
        console.print(f"  ► {fail_rate:.0f}% of runs failed")
    if result.avg_tokens > 0:
        console.print(f"  ► Average context usage: {result.avg_tokens:,.0f} tokens/run")

    # Run table
    if result.runs:
        console.print("\n[bold]Recent Runs[/]")
        table = Table(box=box.SIMPLE)
        table.add_column("ID", style="bold")
        table.add_column("Name")
        table.add_column("Tokens", justify="right")
        table.add_column("Status")
        table.add_column("Latency", justify="right")
        table.add_column("Age", style="dim")

        for r in result.runs[:10]:
            tokens = sum(c.input_tokens + c.output_tokens for c in r.calls)
            latency = sum(c.latency_ms for c in r.calls)
            table.add_row(
                r.id,
                r.name or "—",
                f"{tokens:,}",
                _status_icon(r.status),
                f"{latency:,}ms",
                _format_age(r.started_at),
            )
        console.print(table)
