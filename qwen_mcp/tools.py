"""Tool schemas + handlers for the agent loop.

Every handler takes (ctx, args) and returns a JSON-serializable dict.
File handlers route paths through sandbox.safe_resolve. run_command does not.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qwen_mcp.sandbox import safe_resolve

MAX_RESULT_BYTES = 8 * 1024


@dataclass
class ToolContext:
    """Per-delegation state passed to every handler."""
    working_dir: Path
    files_changed: set[str] = field(default_factory=set)
    commands_run: list[str] = field(default_factory=list)


# ---- Read tools --------------------------------------------------------------

def read_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = safe_resolve(ctx.working_dir, args["path"])
    raw = path.read_bytes()
    total = len(raw)

    offset = args.get("offset", 0)
    limit = args.get("limit")
    chunk = raw[offset:] if limit is None else raw[offset:offset + limit]

    truncated = False
    if len(chunk) > MAX_RESULT_BYTES:
        chunk = chunk[:MAX_RESULT_BYTES]
        truncated = True

    text = chunk.decode("utf-8", errors="replace")
    if limit is None and offset == 0 and total > MAX_RESULT_BYTES:
        truncated = True
    return {"content": text, "truncated": truncated, "total_bytes": total}


def list_dir(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = safe_resolve(ctx.working_dir, args["path"])
    entries = []
    for child in sorted(path.iterdir()):
        kind = "dir" if child.is_dir() else "file" if child.is_file() else "other"
        size = child.stat().st_size if child.is_file() else None
        entries.append({"name": child.name, "type": kind, "size": size})
    return {"entries": entries}


def glob_(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    pattern = args["pattern"]
    root = ctx.working_dir.resolve()
    matches = []
    for match in root.glob(pattern):
        try:
            rel = match.resolve().relative_to(root)
        except ValueError:
            continue  # symlinks pointing outside
        matches.append(str(rel))
    return {"matches": sorted(matches)}


# ---- Write tools -------------------------------------------------------------
# (filled in next task)

def write_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def edit_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def run_command(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


# ---- Schemas + dispatcher ----------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Returns up to ~8KB; use offset/limit for paging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory (one level).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern under the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. 'src/**/*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent dirs if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace `old` with `new` in a file. Errors if `old` is not unique unless replace_all=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                },
                "required": ["command"],
            },
        },
    },
]


_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "glob": glob_,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
}


def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name. Errors are returned to the caller as structured dicts.

    The agent loop turns these into a `tool` message so Qwen can self-correct.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(ctx, args)
    except Exception as e:  # noqa: BLE001 — intentional broad catch
        return {"error": f"{type(e).__name__}: {e}"}
