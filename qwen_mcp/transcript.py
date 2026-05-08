"""JSONL transcript writer for delegations."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

DELEGATIONS_DIR = ".qwen-delegations"


class Transcript:
    """Append-only JSONL log for one delegation."""

    def __init__(self, path: Path, fh: TextIO) -> None:
        self.path = path
        self._fh = fh

    @classmethod
    def open(cls, working_dir: Path) -> "Transcript":
        directory = working_dir / DELEGATIONS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        ensure_gitignore_entry(working_dir, f"{DELEGATIONS_DIR}/")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        short = uuid.uuid4().hex[:8]
        path = directory / f"{ts}-{short}.jsonl"
        fh = path.open("a", encoding="utf-8")
        return cls(path, fh)

    def append(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def ensure_gitignore_entry(working_dir: Path, entry: str) -> None:
    """If a .gitignore exists at working_dir, ensure `entry` is in it.

    Does not create a .gitignore if the project doesn't already have one.
    """
    gitignore = working_dir / ".gitignore"
    if not gitignore.exists():
        return
    contents = gitignore.read_text()
    lines = {line.strip() for line in contents.splitlines()}
    if entry.strip() in lines:
        return
    suffix = "" if contents.endswith("\n") else "\n"
    gitignore.write_text(contents + suffix + entry + "\n")


def _json_default(o: Any) -> Any:
    """Serialize objects pydantic/openai may hand us."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
