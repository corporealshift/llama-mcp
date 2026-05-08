"""Tests for the tool dispatcher and individual handlers."""
import json
from pathlib import Path

import pytest

from qwen_mcp.sandbox import SandboxEscape
from qwen_mcp.tools import (
    TOOL_SCHEMAS,
    ToolContext,
    edit_file,
    glob_,
    list_dir,
    read_file,
    run_command,
    write_file,
)


@pytest.fixture
def ctx(working_dir: Path) -> ToolContext:
    return ToolContext(working_dir=working_dir)


def test_read_file_returns_content(working_dir: Path, ctx: ToolContext):
    (working_dir / "hello.txt").write_text("hi there")
    out = read_file(ctx, {"path": "hello.txt"})
    assert out["content"] == "hi there"
    assert out["truncated"] is False
    assert out["total_bytes"] == 8


def test_read_file_truncates_large_content(working_dir: Path, ctx: ToolContext):
    big = "x" * 20_000
    (working_dir / "big.txt").write_text(big)
    out = read_file(ctx, {"path": "big.txt"})
    assert out["truncated"] is True
    assert len(out["content"]) <= 8 * 1024
    assert out["total_bytes"] == 20_000


def test_read_file_offset_and_limit(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("0123456789")
    out = read_file(ctx, {"path": "f.txt", "offset": 3, "limit": 4})
    assert out["content"] == "3456"


def test_read_file_sandbox_escape_raises(working_dir: Path, ctx: ToolContext):
    with pytest.raises(SandboxEscape):
        read_file(ctx, {"path": "../escaped.txt"})


def test_list_dir_returns_entries(working_dir: Path, ctx: ToolContext):
    (working_dir / "a.txt").write_text("x")
    (working_dir / "sub").mkdir()
    out = list_dir(ctx, {"path": "."})
    names = {e["name"] for e in out["entries"]}
    assert names == {"a.txt", "sub"}
    types = {e["name"]: e["type"] for e in out["entries"]}
    assert types["a.txt"] == "file"
    assert types["sub"] == "dir"


def test_glob_matches_pattern(working_dir: Path, ctx: ToolContext):
    (working_dir / "a.py").write_text("")
    (working_dir / "b.py").write_text("")
    (working_dir / "c.txt").write_text("")
    out = glob_(ctx, {"pattern": "*.py"})
    assert sorted(out["matches"]) == ["a.py", "b.py"]


def test_tool_schemas_have_required_shape():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "read_file", "list_dir", "glob",
        "write_file", "edit_file", "run_command",
    }
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
