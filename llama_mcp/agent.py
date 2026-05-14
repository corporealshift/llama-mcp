"""The agent loop: one delegation start to finish."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llama_mcp.tools import MAX_RESULT_BYTES, TOOL_SCHEMAS, ToolContext, dispatch
from llama_mcp.transcript import Transcript

SYSTEM_PROMPT = """\
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
"""


class ChatClient(Protocol):
    def chat_completions(self, *, messages, tools, tool_choice="auto") -> Any: ...


@dataclass
class AgentResult:
    result: str
    files_changed: list[str]
    commands_run: list[str]
    steps: int
    stop_reason: str
    transcript_path: str


def run_delegation(
    *,
    client: ChatClient,
    working_dir: Path,
    task: str,
    context_hints: list[str],
    max_steps: int,
    timeout_seconds: int,
    max_tokens_total: int,
) -> AgentResult:
    ctx = ToolContext(working_dir=working_dir)
    transcript = Transcript.open(working_dir)
    transcript.append({"type": "meta", "task": task, "working_dir": str(working_dir),
                       "context_hints": context_hints,
                       "max_steps": max_steps, "timeout_seconds": timeout_seconds,
                       "max_tokens_total": max_tokens_total})

    user_content = f"Working directory: {working_dir}\n"
    if context_hints:
        user_content += f"Files worth looking at first: {context_hints}\n"
    user_content += f"\nTask:\n{task}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_assistant_text: str = ""
    tokens_used = 0
    step = 0
    deadline = time.monotonic() + timeout_seconds

    try:
        while True:
            if step >= max_steps:
                stop_reason = "max_steps"; break
            if time.monotonic() >= deadline:
                stop_reason = "timeout"; break
            if tokens_used >= max_tokens_total:
                stop_reason = "token_limit"; break

            step += 1
            t0 = time.monotonic()
            try:
                resp = client.chat_completions(
                    messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto",
                )
            except Exception as e:  # noqa: BLE001
                transcript.append({"step": step, "type": "error",
                                   "where": "chat_completions", "error": str(e)})
                _log(f"step={step} error={type(e).__name__}: {e}")
                stop_reason = "error"
                last_assistant_text = f"OpenAI client error: {e}"
                break

            usage = getattr(resp, "usage", None)
            if usage is not None:
                tokens_used += getattr(usage, "total_tokens", 0) or 0

            msg = resp.choices[0].message
            transcript.append({"step": step, "type": "assistant",
                               "message": _to_jsonable(msg)})
            messages.append(_assistant_to_message(msg))

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                last_assistant_text = msg.content or ""
                stop_reason = "complete"
                _log(f"step={step} complete dur_ms={int((time.monotonic()-t0)*1000)}")
                break

            last_assistant_text = msg.content or last_assistant_text

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as e:
                    args = None
                    result = {"error": f"invalid JSON in arguments: {e}"}
                else:
                    result = dispatch(ctx, name, args)

                transcript.append({"step": step, "type": "tool",
                                   "tool_call_id": call.id, "name": name,
                                   "args": args, "result": result})
                ok = "error" not in result
                _log(f"step={step} tool={name} ok={ok}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _truncate_json(result, MAX_RESULT_BYTES),
                })
        else:
            stop_reason = "complete"  # unreachable; loop exits via break
    finally:
        transcript.append({"type": "meta", "stop_reason": locals().get("stop_reason", "error"),
                           "steps": step, "tokens_used": tokens_used})
        transcript.close()

    return AgentResult(
        result=last_assistant_text,
        files_changed=sorted(ctx.files_changed),
        commands_run=list(ctx.commands_run),
        steps=step,
        stop_reason=stop_reason,
        transcript_path=str(transcript.path),
    )


def _log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _to_jsonable(msg: Any) -> Any:
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {"content": getattr(msg, "content", None)}


def _assistant_to_message(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI assistant message back into a dict suitable for re-sending."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


def _truncate_json(obj: Any, max_bytes: int) -> str:
    s = json.dumps(obj, default=str)
    if len(s.encode("utf-8")) <= max_bytes:
        return s
    truncated = s.encode("utf-8")[:max_bytes].decode("utf-8", "replace")
    return truncated + f"\n[truncated, full size {len(s)} bytes]"
