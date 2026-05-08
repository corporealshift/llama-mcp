# qwen-mcp — Design

**Date:** 2026-05-08
**Author:** Kyle Diedrick (with Claude)
**Status:** Draft, pending implementation plan

## Problem

Claude Code is a strong orchestrator but every token of inference goes to a paid API. Many code-writing subtasks are mechanical enough that a smaller local model can handle them: writing a function, refactoring within a file, adding tests, threading a parameter through. Kyle has GPU capacity for Qwen 3.6 running locally via `llama.cpp`. We want Claude Code to be able to *delegate* discrete subtasks to local Qwen and receive a result, treating Qwen as a subagent.

## Goals

- Claude Code can hand a self-contained coding task to local Qwen and get back a result describing what was done, with the file changes already on disk.
- Qwen operates as a real agent — exploring, reading, editing, running commands — not just emitting code text.
- All filesystem and shell access happens in WSL, where the project actually lives. Qwen on Windows never directly touches files; it requests tool calls and the MCP server (in WSL) executes them.
- Sandbox per delegation: file operations are rooted at a caller-provided working directory and cannot escape it.
- The full agent loop is observable both live (stderr) and after the fact (JSONL transcript on disk).

## Non-goals

- Streaming Qwen's intermediate reasoning back into Claude Code's context. MCP tools are request/response; observability lives in stderr and transcripts instead.
- Letting Qwen drive multiple, concurrent subagents. One delegation at a time per call.
- Replacing Claude Code's own filesystem tools. Claude is still in charge of orchestration; Qwen is a worker called when Claude chooses.
- Caching, rate-limiting, or multi-tenant concerns — this is a personal tool talking to a single local server.
- Project-aware "memory" persistence across delegations beyond the on-disk transcripts.

## Architecture

```
┌────────────────────┐  stdio (MCP)  ┌──────────────────────────────────┐
│   Claude Code      │◄────────────►│   qwen-mcp server  (Python)      │
│   (WSL2 Ubuntu)    │               │                                  │
└────────────────────┘               │  ┌───────────────────────────┐   │
                                     │  │ MCP layer (FastMCP)       │   │
                                     │  │  • tool: delegate_to_qwen │   │
                                     │  └────────────┬──────────────┘   │
                                     │               │                  │
                                     │  ┌────────────▼──────────────┐   │
                                     │  │ Agent loop                │   │
                                     │  │  • build OpenAI request   │   │
                                     │  │  • dispatch tool calls    │   │
                                     │  │  • track step/time/tokens │   │
                                     │  │  • write transcript       │   │
                                     │  └──┬──────────────────────┬─┘   │
                                     │     │                      │     │
                                     │  ┌──▼─────────────┐   ┌────▼──┐  │
                                     │  │ Sandboxed tool │   │ HTTP  │  │
                                     │  │ executor       │   │ to    │  │
                                     │  │ (cwd=root)     │   │ Qwen  │  │
                                     │  └────────────────┘   └───┬───┘  │
                                     └──────────────────────────┼──────┘
                                                                │
                                                  HTTP POST /v1/chat/completions
                                                  to <win_host>:8033
                                                                │
                                                                ▼
                                              ┌───────────────────────────┐
                                              │  llama.cpp llama-server   │
                                              │  Qwen 3.6 (Windows host)  │
                                              │  --host 0.0.0.0 --jinja   │
                                              └───────────────────────────┘
```

### Code layout

Project root: `/home/kyle/qwen-mcp/` (separate Git repository, sibling to `lopress`).

Installed via `pipx install -e .` so `~/.local/bin/qwen-mcp` re-runs the source on every launch — no symlink maintenance, no reinstall while iterating.

```
qwen-mcp/
├── pyproject.toml           # entry point: qwen-mcp = qwen_mcp.server:main
├── README.md
├── docs/superpowers/specs/  # this spec moves here once the repo is initialized
├── qwen_mcp/
│   ├── __init__.py
│   ├── server.py            # FastMCP entry; registers delegate_to_qwen
│   ├── agent.py             # the loop: budgets, dispatch, transcript
│   ├── openai_client.py     # OpenAI client + WSL→Windows host resolution
│   ├── tools.py             # tool schemas + dispatcher
│   ├── sandbox.py           # safe_resolve(working_dir, user_path)
│   ├── transcript.py        # JSONL writer
│   └── config.py            # env loading + defaults
├── tests/
│   ├── test_sandbox.py
│   ├── test_tools.py
│   ├── test_agent.py        # uses fake OpenAI client
│   ├── test_config.py
│   └── test_server.py
└── scripts/
    └── smoke.py             # manual end-to-end against real llama-server
```

## The MCP tool

A single tool exposed to Claude Code:

```python
delegate_to_qwen(
    task: str,                          # what to do, written by Claude
    working_dir: str,                   # absolute path; sandbox root
    context_hints: list[str] = [],      # files Qwen should look at first
    max_steps: int | None = None,       # override default 45
    timeout_seconds: int | None = None, # override default 1800
    max_tokens_total: int | None = None # override default 200_000
) -> dict
```

Returns:

```python
{
    "result": str,                  # Qwen's last assistant content; if the loop
                                    # stopped early, this is whatever Qwen had
                                    # said most recently (possibly empty)
    "files_changed": list[str],     # tracked at the tool layer
    "commands_run": list[str],      # commands passed to run_command
    "steps": int,                   # iterations consumed
    "stop_reason": str,             # complete | max_steps | timeout | token_limit | error
    "transcript_path": str          # absolute path to the JSONL transcript
}
```

## Qwen's view: system prompt and tools

System prompt (kept short — Qwen's context budget matters):

```
You are a code-writing subagent. A more capable orchestrator delegated this
task to you. Work autonomously inside the given working directory.

Rules:
- Stay inside the working directory. Do not access paths outside it.
- Read before you write. Use list_dir/read_file/glob to understand the code
  before editing.
- Make the smallest change that satisfies the task. Do not refactor unrelated
  code. Do not add comments unless they explain non-obvious "why".
- Use run_command for builds, tests, formatters when relevant. Treat command
  failures as information, not as instructions to retry blindly.
- When done, reply with a concise summary (what you changed, which files,
  any tests run and their results). Do not include code blocks of full files
  in the summary — the orchestrator can read the diffs.

Available tools: read_file, list_dir, glob, write_file, edit_file, run_command.
```

Tool schemas (OpenAI function-calling format):

| Tool          | Params                                                           | Returns                                       | Notes |
|---------------|------------------------------------------------------------------|-----------------------------------------------|-------|
| `read_file`   | `path`, `offset?`, `limit?`                                      | `{content, truncated, total_bytes}`           | Truncated to 8 KB by default; `offset`/`limit` for paging large files |
| `list_dir`    | `path`                                                           | `{entries: [{name, type, size}]}`             | One level only |
| `glob`        | `pattern`                                                        | `{matches: [str]}`                            | Resolved relative to `working_dir` |
| `write_file`  | `path`, `content`                                                | `{bytes_written}`                             | Creates parent dirs; appends `path` to `files_changed` |
| `edit_file`   | `path`, `old`, `new`, `replace_all?`                             | `{replacements}`                              | Errors if `old` not unique and `replace_all=false`; appends `path` to `files_changed` |
| `run_command` | `command`, `timeout?` (default 120s, max 600s)                   | `{stdout, stderr, exit_code, truncated, duration_ms, timed_out}` | Run via `bash -lc`, `cwd=working_dir`; stdout/stderr capped at 8 KB each in tool reply (full in transcript); appends `command` to `commands_run` |

### Sandboxing

`sandbox.safe_resolve(working_dir, user_path) -> Path`:

1. If `user_path` is relative, resolve it against `working_dir`; if absolute, use as-is.
2. Call `.resolve()` to follow symlinks and normalize `..`.
3. Assert `working_dir.resolve()` is a parent of the result; raise `SandboxEscape` if not.
4. All file tools (`read_file`, `list_dir`, `glob`, `write_file`, `edit_file`) route paths through this.
5. `SandboxEscape` is converted to a structured tool error returned to Qwen, not a loop crash — Qwen self-corrects.

`run_command` deliberately does *not* sandbox — full shell access within `cwd=working_dir` is allowed by design. Qwen can `cd /` from there if it wants; that's an accepted risk we may tighten later.

## Agent loop

```
1. server.py validates inputs (working_dir exists, is absolute, etc.)
2. agent.py opens transcript at <working_dir>/.qwen-delegations/<ts>-<uuid>.jsonl
   and ensures .qwen-delegations/ is in the project's .gitignore (auto-add on
   first write if absent).
3. Build initial messages:
   [
     {role: "system", content: SYSTEM_PROMPT},
     {role: "user", content: f"Working directory: {working_dir}\n"
                              f"Files worth looking at first: {context_hints}\n\n"
                              f"Task:\n{task}"}
   ]
4. Loop:
     while True:
       check budgets (steps, time, tokens) → break with stop_reason on trip
       resp = openai_client.chat_completions(messages, tools=TOOL_SCHEMAS,
                                              tool_choice="auto")
       msg = resp.choices[0].message
       transcript.append({step, "assistant", msg})
       messages.append(msg)
       if not msg.tool_calls:
           stop_reason = "complete"; break
       for call in msg.tool_calls:
           result = tools.dispatch(call, working_dir)
           transcript.append({step, "tool", call.name, result})
           messages.append({role: "tool", tool_call_id: call.id,
                            content: json.dumps(truncate(result, 8KB))})
5. Return the structured response (see "The MCP tool" above).
```

`files_changed` is tracked at the tool layer — `write_file` and `edit_file` each append their resolved path. `run_command` cannot reliably know which files it touched, so it does *not* contribute to `files_changed`; it contributes to `commands_run` instead.

Tool results are truncated to ~8 KB per result before being appended to `messages`, with a `[truncated, N more bytes]` marker. The transcript stores the untruncated result.

## Configuration

Loaded from environment at server start; per-call args override defaults.

| Var                           | Default                                       | Purpose |
|-------------------------------|-----------------------------------------------|---------|
| `QWEN_BASE_URL`               | `http://host.docker.internal:8033/v1`         | If hostname doesn't resolve or is `localhost`/`127.0.0.1`, `config.py` substitutes the WSL2 default-route gateway IP. |
| `QWEN_MODEL`                  | `qwen`                                        | Sent in the request; llama-server ignores it but the field is required. |
| `QWEN_API_KEY`                | `sk-no-key`                                   | llama-server doesn't auth, but the OpenAI client requires non-empty. |
| `QWEN_DEFAULT_MAX_STEPS`      | `45`                                          | |
| `QWEN_DEFAULT_TIMEOUT_SECONDS`| `1800`                                        | |
| `QWEN_DEFAULT_MAX_TOKENS_TOTAL`| `200000`                                     | Hard ceiling across the whole loop. |
| `QWEN_LOG_LEVEL`              | `INFO`                                        | |

Claude Code MCP registration (per-project `.mcp.json` or user `~/.claude.json`):

```json
{
  "mcpServers": {
    "qwen": {
      "command": "qwen-mcp",
      "env": { "QWEN_BASE_URL": "http://host.docker.internal:8033/v1" }
    }
  }
}
```

## Error handling

| Error                                        | Where                | Behavior |
|----------------------------------------------|----------------------|----------|
| llama-server unreachable / connection error  | `openai_client`      | Retry 3× with exponential backoff (1s/2s/4s). Final failure → `stop_reason="error"` with descriptive message. |
| HTTP 5xx                                     | `openai_client`      | Same retry policy. |
| HTTP 4xx                                     | `openai_client`      | No retry — request is malformed (bug). Bubble up as `error`. |
| Malformed `tool_calls` JSON args             | `tools.dispatch`     | Structured error returned to Qwen as the tool result. Loop continues. |
| Unknown tool name                            | `tools.dispatch`     | Same. |
| `SandboxEscape`                              | `tools.dispatch`     | Same. |
| `run_command` exceeds inner timeout          | `tools.run_command`  | Kill process; return `exit_code=-1, timed_out=true` to Qwen. |
| Whole-loop budget hit                        | `agent`              | Clean exit with `stop_reason ∈ {max_steps, timeout, token_limit}`; partial state returned. |
| Qwen returns no `tool_calls` and no content  | `agent`              | Treat as `complete` with empty result. Rare. |

## Observability

- **stderr** — one structured line per step: `step=N tool=read_file dur=12ms ok=true` (or `ok=false err=<short>`). Claude Code surfaces MCP server stderr in its UI.
- **JSONL transcript** — `<working_dir>/.qwen-delegations/<ISO-timestamp>-<short-uuid>.jsonl`, one JSON object per line: `{step, type: "assistant"|"tool"|"meta", ...}`. Untruncated. The directory is auto-created on first delegation; `.qwen-delegations/` is auto-added to `.gitignore` if a `.gitignore` exists at `working_dir`.

## Testing

| Layer            | What we test                                                                                      | How |
|------------------|---------------------------------------------------------------------------------------------------|-----|
| `sandbox.py`     | `safe_resolve` rejects `../`, absolute paths outside root, symlinks pointing out, edges          | Unit, `tmp_path`. |
| `tools.py`       | Each handler: happy path, sandbox escape, missing file, truncation; `run_command` timeout         | Unit, `tmp_path`. Portable shell commands (`echo`, `false`). |
| `agent.py`       | Step counter, timeout, token cap, each `stop_reason`, malformed tool-call recovery, accounting    | Fake OpenAI client returning scripted responses. No real HTTP. |
| `config.py`      | Env loading, gateway resolution, URL rewrite for `localhost`/`127.0.0.1`                          | Unit; mock `subprocess`/`socket`. |
| `server.py`      | Tool registration, schema validation rejects bad inputs                                           | FastMCP in-process test client. |
| End-to-end smoke | One real round-trip against actual llama-server                                                   | `scripts/smoke.py`, manual; not in CI. |

Stack: `pytest` + `pytest-asyncio` (FastMCP is async). Coverage target: every error branch in `agent.py` and every escape path in `sandbox.py` has a dedicated test.

## WSL2 ↔ Windows networking

Captured here because the gotchas are easy to forget:

- WSL2 on Windows 10 cannot use `localhost` to reach Windows services (no mirrored networking — that's Win11 22H2+ only). Use the WSL2 default-route gateway IP. `config.py` resolves this automatically when the configured `QWEN_BASE_URL` host is `localhost` or `127.0.0.1`, or when DNS for the configured host fails.
- `llama-server` must be launched with `--host 0.0.0.0 --jinja`. The `--jinja` flag is what makes Qwen's tool-call format parse into the `tool_calls` field; without it, the server returns plain text and the agent loop will never get a tool call.
- A Windows Defender Firewall inbound rule must allow TCP 8033 (Profile Any — WSL2's vEthernet adapter is typically classified Public, not Private).
- Verified working as of 2026-05-08: network reachability and OpenAI tool-calling round-trip both succeed.

## Open questions

- Do we want a `dry_run` flag on `delegate_to_qwen` that runs the full loop but rejects all `write_file`/`edit_file`/`run_command` calls (turning them into structured errors Qwen sees)? Useful for "what would Qwen do?" before letting it actually do it. Could be added in a follow-up if the basic flow proves valuable.
- Shell sandbox tightening (allowlist or denylist) is deferred — start freeform, revisit if it becomes a problem in practice.
