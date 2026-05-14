"""Tests for sandbox.safe_resolve."""
from pathlib import Path

import pytest

from llama_mcp.sandbox import SandboxEscape, safe_resolve


def test_relative_path_resolves_under_root(working_dir: Path):
    (working_dir / "foo.txt").write_text("x")
    result = safe_resolve(working_dir, "foo.txt")
    assert result == (working_dir / "foo.txt").resolve()


def test_absolute_path_inside_root_allowed(working_dir: Path):
    (working_dir / "foo.txt").write_text("x")
    inside = str(working_dir / "foo.txt")
    result = safe_resolve(working_dir, inside)
    assert result == (working_dir / "foo.txt").resolve()


def test_dotdot_escape_rejected(working_dir: Path):
    with pytest.raises(SandboxEscape):
        safe_resolve(working_dir, "../escaped.txt")


def test_absolute_path_outside_root_rejected(working_dir: Path):
    with pytest.raises(SandboxEscape):
        safe_resolve(working_dir, "/etc/passwd")


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="symlink creation requires elevated privileges on Windows",
)
def test_symlink_pointing_outside_rejected(working_dir: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("nope")
    link = working_dir / "link"
    link.symlink_to(outside)
    with pytest.raises(SandboxEscape):
        safe_resolve(working_dir, "link")


def test_nonexistent_path_under_root_allowed(working_dir: Path):
    """Writing a new file means the path doesn't exist yet — must still resolve."""
    result = safe_resolve(working_dir, "subdir/new.txt")
    assert result == (working_dir / "subdir/new.txt").resolve()


def test_root_itself_allowed(working_dir: Path):
    result = safe_resolve(working_dir, ".")
    assert result == working_dir.resolve()


def test_trailing_slash_handled(working_dir: Path):
    (working_dir / "sub").mkdir()
    result = safe_resolve(working_dir, "sub/")
    assert result == (working_dir / "sub").resolve()


def test_working_dir_must_be_absolute():
    with pytest.raises(ValueError):
        safe_resolve(Path("relative/dir"), "foo.txt")
