# starnose

[![npm](https://img.shields.io/npm/v/snose?color=c4607a&label=npm)](https://www.npmjs.com/package/snose)
[![PyPI](https://img.shields.io/pypi/v/starnose?color=c4607a&label=pypi)](https://pypi.org/project/starnose/)
[![License: MIT](https://img.shields.io/badge/license-MIT-c4607a)](LICENSE)

See exactly what Claude Code is reading, thinking, and doing.
Every token, every skill load, every compaction event — live at the network layer.

```bash
# npm
npx snose on

# pip
pip install starnose && snose on
```

## commands

| command | what it does |
|---|---|
| `snose on` | start proxy daemon on :3399, set `ANTHROPIC_BASE_URL` |
| `snose off` | stop daemon, clear env var |
| `snose status` | show running state, call count, session cost |
| `snose sense` | live feed — every call as it happens, loop detection, compaction alerts |
| `snose dig` | keyboard-driven inspector — expand any call, see token breakdown, search history |

Both `snose` and `starnose` work as binary names.

## how it works

Starnose is an HTTP proxy that sits between Claude Code and the Anthropic API.
One environment variable (`ANTHROPIC_BASE_URL=http://localhost:3399`), zero code changes.
Every request and response is recorded to a local SQLite database with full system prompt parsing,
skill token breakdown, compaction detection, and missing context tracking.

Nothing leaves your machine.

## install

**npm (no install needed):**
```bash
npx snose on
```

**pip:**
```bash
pip install starnose
snose on
```

**from source:**
```bash
git clone https://github.com/eitanlebras/starnose
cd starnose
npm install && npm run build
snose on
```

## usage

```bash
snose on                    # start recording
claude "refactor auth"      # work normally
snose sense                 # watch live in another terminal
snose dig                   # inspect sessions after
snose off                   # stop recording
```

---

MIT · [starnose.dev](https://starnose.dev)
[![MCP Server](https://volspan.dev/api/badge/eitanlebras-starnose?v=4)](https://volspan.dev/servers/eitanlebras-starnose)
