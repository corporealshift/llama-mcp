"""Tests for the FastMCP server registration and input validation."""
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen_mcp.server import _validate_inputs, build_server


def test_validate_inputs_accepts_valid(working_dir: Path):
    _validate_inputs(
        task="do a thing",
        working_dir=str(working_dir),
        context_hints=[],
        max_steps=10, timeout_seconds=60, max_tokens_total=10_000,
    )


def test_validate_inputs_rejects_relative_working_dir(working_dir: Path):
    with pytest.raises(ValueError, match="absolute"):
        _validate_inputs(
            task="x", working_dir="relative/path",
            context_hints=[], max_steps=10, timeout_seconds=60,
            max_tokens_total=10_000,
        )


def test_validate_inputs_rejects_missing_working_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        _validate_inputs(
            task="x", working_dir=str(tmp_path / "nope"),
            context_hints=[], max_steps=10, timeout_seconds=60,
            max_tokens_total=10_000,
        )


def test_validate_inputs_rejects_empty_task(working_dir: Path):
    with pytest.raises(ValueError, match="task"):
        _validate_inputs(
            task="   ", working_dir=str(working_dir),
            context_hints=[], max_steps=10, timeout_seconds=60,
            max_tokens_total=10_000,
        )


def test_validate_inputs_rejects_bad_step_count(working_dir: Path):
    with pytest.raises(ValueError, match="max_steps"):
        _validate_inputs(
            task="x", working_dir=str(working_dir),
            context_hints=[], max_steps=0, timeout_seconds=60,
            max_tokens_total=10_000,
        )


def test_build_server_registers_delegate_tool():
    """The server exposes a `delegate_to_qwen` tool."""
    server = build_server()
    # FastMCP exposes _tool_manager._tools (or similar); we test via list_tools.
    tools = server._tool_manager.list_tools()  # internal API; acceptable for a smoke check
    names = [t.name for t in tools]
    assert "delegate_to_qwen" in names
