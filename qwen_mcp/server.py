"""FastMCP entry point that exposes the `delegate_to_qwen` tool."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from qwen_mcp import config as config_module
from qwen_mcp.agent import run_delegation
from qwen_mcp.openai_client import QwenClient


def build_server() -> FastMCP:
    cfg = config_module.load()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(message)s",
        stream=sys.stderr,
    )

    server = FastMCP("qwen-mcp")
    client = QwenClient(cfg)

    @server.tool()
    def delegate_to_qwen(
        task: str,
        working_dir: str,
        context_hints: list[str] | None = None,
        max_steps: int | None = None,
        timeout_seconds: int | None = None,
        max_tokens_total: int | None = None,
    ) -> dict:
        """Delegate a coding subtask to a local Qwen instance.

        Args:
            task: Self-contained description of what Qwen should do.
            working_dir: Absolute path; sandbox root for all file tools.
            context_hints: Files Qwen should look at first (optional).
            max_steps: Override default tool-call rounds.
            timeout_seconds: Override wall-clock cap.
            max_tokens_total: Override total-token cap.

        Returns:
            dict with result, files_changed, commands_run, steps, stop_reason,
            transcript_path.
        """
        ms = max_steps if max_steps is not None else cfg.default_max_steps
        ts = timeout_seconds if timeout_seconds is not None else cfg.default_timeout_seconds
        mt = max_tokens_total if max_tokens_total is not None else cfg.default_max_tokens_total
        hints = list(context_hints or [])

        _validate_inputs(
            task=task, working_dir=working_dir, context_hints=hints,
            max_steps=ms, timeout_seconds=ts, max_tokens_total=mt,
        )

        result = run_delegation(
            client=client,
            working_dir=Path(working_dir),
            task=task,
            context_hints=hints,
            max_steps=ms,
            timeout_seconds=ts,
            max_tokens_total=mt,
        )
        return {
            "result": result.result,
            "files_changed": result.files_changed,
            "commands_run": result.commands_run,
            "steps": result.steps,
            "stop_reason": result.stop_reason,
            "transcript_path": result.transcript_path,
        }

    return server


def _validate_inputs(
    *,
    task: str,
    working_dir: str,
    context_hints: list[str],
    max_steps: int,
    timeout_seconds: int,
    max_tokens_total: int,
) -> None:
    if not task or not task.strip():
        raise ValueError("task must be a non-empty string")

    wd = Path(working_dir)
    if not wd.is_absolute():
        raise ValueError(f"working_dir must be absolute: {working_dir!r}")
    if not wd.exists():
        raise ValueError(f"working_dir does not exist: {working_dir!r}")
    if not wd.is_dir():
        raise ValueError(f"working_dir is not a directory: {working_dir!r}")

    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be >= 1")
    if max_tokens_total < 1:
        raise ValueError("max_tokens_total must be >= 1")

    if not isinstance(context_hints, list):
        raise ValueError("context_hints must be a list of strings")


def main() -> None:
    """Entry point for the `qwen-mcp` console script."""
    server = build_server()
    server.run()  # FastMCP defaults to stdio transport.


if __name__ == "__main__":
    main()
