"""Tests for the tool dispatcher and individual handlers."""
import json
from pathlib import Path

import pytest

from llama_mcp.sandbox import SandboxEscape
from llama_mcp.tools import (
    TOOL_SCHEMAS,
    ToolContext,
    dispatch,
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


def test_write_file_creates_file(working_dir: Path, ctx: ToolContext):
    out = write_file(ctx, {"path": "new.txt", "content": "hello"})
    assert out["bytes_written"] == 5
    assert (working_dir / "new.txt").read_text() == "hello"
    assert "new.txt" in {Path(p).name for p in ctx.files_changed}


def test_write_file_creates_parent_dirs(working_dir: Path, ctx: ToolContext):
    write_file(ctx, {"path": "deep/nested/file.txt", "content": "x"})
    assert (working_dir / "deep/nested/file.txt").read_text() == "x"


def test_write_file_overwrites(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("old")
    write_file(ctx, {"path": "f.txt", "content": "new"})
    assert (working_dir / "f.txt").read_text() == "new"


def test_write_file_sandbox_escape_raises(ctx: ToolContext):
    with pytest.raises(SandboxEscape):
        write_file(ctx, {"path": "../bad.txt", "content": "x"})


def test_edit_file_replaces_unique_string(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("foo bar baz")
    out = edit_file(ctx, {"path": "f.txt", "old": "bar", "new": "qux"})
    assert out["replacements"] == 1
    assert (working_dir / "f.txt").read_text() == "foo qux baz"
    assert "f.txt" in {Path(p).name for p in ctx.files_changed}


def test_edit_file_errors_when_old_not_unique(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("xx xx")
    with pytest.raises(ValueError, match="not unique"):
        edit_file(ctx, {"path": "f.txt", "old": "xx", "new": "yy"})


def test_edit_file_replace_all(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("xx xx")
    out = edit_file(ctx, {
        "path": "f.txt", "old": "xx", "new": "yy", "replace_all": True
    })
    assert out["replacements"] == 2
    assert (working_dir / "f.txt").read_text() == "yy yy"


def test_edit_file_errors_when_old_missing(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("foo")
    with pytest.raises(ValueError, match="not found"):
        edit_file(ctx, {"path": "f.txt", "old": "missing", "new": "x"})


def test_run_command_captures_stdout(ctx: ToolContext):
    out = run_command(ctx, {"command": "python -c \"print('hello')\""})
    assert out["exit_code"] == 0
    assert out["stdout"].strip() == "hello"
    assert out["timed_out"] is False


def test_run_command_captures_stderr_and_exit_code(ctx: ToolContext):
    out = run_command(ctx, {"command": "python -c \"import sys; sys.stderr.write('err\\n'); sys.exit(3)\""})
    assert out["exit_code"] == 3
    assert "err" in out["stderr"]


def test_run_command_runs_in_working_dir(working_dir: Path, ctx: ToolContext):
    (working_dir / "marker").write_text("yes")
    out = run_command(ctx, {"command": "python -c \"import os; print('\\n'.join(os.listdir('.')))\""})
    assert "marker" in out["stdout"]


def test_run_command_truncates_large_output(ctx: ToolContext):
    out = run_command(ctx, {"command": "python -c \"print('x' * 20000)\""})
    assert out["truncated"] is True
    assert len(out["stdout"]) <= 8 * 1024


def test_run_command_times_out(ctx: ToolContext):
    out = run_command(ctx, {"command": "python -c \"import time; time.sleep(10)\"", "timeout": 1})
    assert out["timed_out"] is True
    assert out["exit_code"] == -1


def test_dispatch_unknown_tool(ctx: ToolContext):
    out = dispatch(ctx, "no_such_tool", {})
    assert "error" in out
    assert "unknown tool" in out["error"]


def test_dispatch_converts_sandbox_escape_to_error(ctx: ToolContext):
    out = dispatch(ctx, "read_file", {"path": "../escape"})
    assert "error" in out
    assert "SandboxEscape" in out["error"]


def test_dispatch_happy_path(working_dir: Path, ctx: ToolContext):
    (working_dir / "f.txt").write_text("ok")
    out = dispatch(ctx, "read_file", {"path": "f.txt"})
    assert out.get("content") == "ok"
    assert "error" not in out
