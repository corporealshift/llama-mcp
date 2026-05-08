"""Tests for the JSONL transcript writer."""
import json
from pathlib import Path

from qwen_mcp.transcript import Transcript, ensure_gitignore_entry


def test_transcript_creates_directory(working_dir: Path):
    t = Transcript.open(working_dir)
    assert (working_dir / ".qwen-delegations").is_dir()
    t.close()


def test_transcript_filename_has_timestamp_and_uuid(working_dir: Path):
    t = Transcript.open(working_dir)
    name = t.path.name
    assert name.endswith(".jsonl")
    # ISO timestamp + dash + 8-hex chars + .jsonl
    assert "-" in name
    t.close()


def test_transcript_appends_jsonl(working_dir: Path):
    t = Transcript.open(working_dir)
    t.append({"step": 1, "type": "assistant", "content": "hi"})
    t.append({"step": 1, "type": "tool", "name": "read_file", "result": {"ok": True}})
    t.close()

    lines = t.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "assistant"
    assert json.loads(lines[1])["name"] == "read_file"


def test_ensure_gitignore_entry_adds_when_missing(working_dir: Path):
    gitignore = working_dir / ".gitignore"
    gitignore.write_text("__pycache__/\n")
    ensure_gitignore_entry(working_dir, ".qwen-delegations/")
    contents = gitignore.read_text()
    assert ".qwen-delegations/" in contents


def test_ensure_gitignore_entry_idempotent(working_dir: Path):
    gitignore = working_dir / ".gitignore"
    gitignore.write_text("__pycache__/\n.qwen-delegations/\n")
    ensure_gitignore_entry(working_dir, ".qwen-delegations/")
    # File should still contain only one occurrence
    assert gitignore.read_text().count(".qwen-delegations/") == 1


def test_ensure_gitignore_entry_skips_when_no_gitignore(working_dir: Path):
    """Don't create a .gitignore where none exists."""
    ensure_gitignore_entry(working_dir, ".qwen-delegations/")
    assert not (working_dir / ".gitignore").exists()
