# starnose

Your agent is running in the dark. starnose turns the lights on.

`htop` for your agent's context window. Intercept, record, visualize, and compare the context sent to LLMs across agent runs — zero code changes required.

![demo](demo.gif)

## Install

```bash
pip install starnose
```

## Quick Start

```bash
# 1. Run your agent with context recording
snose run python my_agent.py

# 2. Inspect what your agent sent to the LLM
snose inspect

# 3. Compare two runs
snose diff
```

That's it. No code changes. No config files. No API keys.

## How It Works

starnose starts a local proxy that intercepts all OpenAI and Anthropic API calls from your agent. It records every message, token count, and response to a local SQLite database. You then use the TUI tools to inspect, compare, and optimize your agent's context window usage.

## Works With

- **Claude Code** — `snose run -- claude`
- **Codex CLI** — `snose run -- codex`
- **LangChain** — callback handler included
- **OpenAI SDK** — monkeypatch or proxy
- **Anthropic SDK** — monkeypatch or proxy
- **Any OpenAI-compatible agent** — `snose run -- <command>`

## Commands

| Command | Description |
|---------|-------------|
| `snose run <cmd>` | Run a command with context interception |
| `snose inspect [id]` | Inspect a run's context window (TUI) |
| `snose diff [a] [b]` | Compare two runs side-by-side |
| `snose watch` | Live-monitor a running agent |
| `snose stats` | Aggregate stats across runs |
| `snose optimize [id]` | Get optimization suggestions |
| `snose ls` | List recorded runs |
| `snose export [id]` | Export run data to JSON |
| `snose config` | View/modify configuration |

### snose run

```bash
snose run python my_agent.py
snose run --name "gpt4-chunked" --tag prod python agent.py
snose run --compare python agent.py    # auto-diff against last run
```

Flags:
- `--name` — human-readable name
- `--tag` — repeatable tags
- `--model` — override model for token counting
- `--compare` — auto-diff when complete
- `--no-proxy` — disable interception

### snose inspect

```bash
snose inspect              # most recent run
snose inspect run_a2f3     # specific run
snose inspect --last       # pick from recent runs
```

Two-panel TUI showing run metadata, context budget bar, segment breakdown, and full message details.

### snose diff

```bash
snose diff                      # last two runs
snose diff run_a2f3 run_9c81    # specific runs
snose diff --last               # pick two runs
```

Shows segment-level delta table and auto-generated hypotheses explaining performance differences.

### snose watch

```bash
snose watch    # attach to running agent
```

Live-streaming view of context changes as your agent runs.

## Python SDK

```python
from starnose import trace, snapshot

@trace(name="my-run", tags=["prod"])
def my_agent(query: str) -> str:
    ...

# Or as context manager
with trace("experiment-a") as run:
    result = agent.run(query)
    run.snapshot("pre-retrieval", messages)
```

## Integrations

```python
# OpenAI monkeypatch
from starnose.integrations import patch_openai
patch_openai()

# Anthropic monkeypatch
from starnose.integrations import patch_anthropic
patch_anthropic()

# LangChain callback
from starnose.integrations import LangChainTracer
agent = AgentExecutor(..., callbacks=[LangChainTracer()])
```

## Proxy Chaining

If your agent already uses a proxy (e.g. LiteLLM):

```bash
STARNOSE_UPSTREAM=http://localhost:4000 snose run python agent.py
```

## Philosophy

- **Local-first** — all data stays in `~/.starnose/runs.db`
- **Zero code changes** — proxy-based interception, just wrap your command
- **Pipe-friendly** — every command supports `--json` for scripting
- **Never breaks your agent** — proxy errors fail open, always

## Configuration

```bash
snose config                          # show current config
snose config set model gpt-4o         # set default model
snose config set context_limit 128000 # override context limit
snose config reset                    # restore defaults
```

Config stored in `~/.starnose/config.json`.
DB stored in `~/.starnose/runs.db` (override with `STARNOSE_DB` env var).
