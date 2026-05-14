"""Path safety: resolve user-supplied paths to absolute paths under a root.

Every file tool in tools.py routes paths through safe_resolve so Qwen cannot
read or write outside the caller-supplied working directory.
"""
from __future__ import annotations

from pathlib import Path


class SandboxEscape(Exception):
    """Raised when a resolved path escapes the working directory."""


def safe_resolve(working_dir: Path, user_path: str) -> Path:
    """Resolve `user_path` to an absolute Path that lies under `working_dir`.

    - Relative paths are resolved against `working_dir`.
    - Symlinks and `..` are normalized via Path.resolve().
    - The working directory must be absolute.
    - For paths whose target does not yet exist (e.g. write_file targets), the
      check still works because Path.resolve() resolves the longest existing
      prefix and then appends the unresolved tail.

    Raises:
        ValueError: if working_dir is not absolute.
        SandboxEscape: if the resolved path is not under working_dir.
    """
    if not working_dir.is_absolute():
        raise ValueError(f"working_dir must be absolute: {working_dir}")

    root = working_dir.resolve()
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = working_dir / candidate
    resolved = candidate.resolve()

    if resolved != root and root not in resolved.parents:
        raise SandboxEscape(
            f"Path {user_path!r} resolves to {resolved}, which is outside {root}"
        )
    return resolved
