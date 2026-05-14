#!/usr/bin/env python3
"""Manual end-to-end smoke test against a running llama-server.

Requires:
  - llama-server running on the configured LLAMA_BASE_URL with a model GGUF
    loaded and --jinja enabled.

Usage:
  .venv\\Scripts\\activate
  python scripts/smoke.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from llama_mcp.agent import run_delegation
from llama_mcp.config import load
from llama_mcp.openai_client import LlamaClient


def main() -> int:
    cfg = load()
    print(f"Using base_url={cfg.base_url} model={cfg.model}", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="llama-smoke-"))
    try:
        client = LlamaClient(cfg)
        result = run_delegation(
            client=client,
            working_dir=workdir,
            task=(
                "Create a file called hello.txt in the working directory whose "
                "single line of content is the word: hi"
            ),
            context_hints=[],
            max_steps=15,
            timeout_seconds=300,
            max_tokens_total=50_000,
        )

        print(f"stop_reason={result.stop_reason} steps={result.steps}")
        print(f"files_changed={result.files_changed}")
        print(f"transcript={result.transcript_path}")
        print(f"--- result ---\n{result.result}")

        target = workdir / "hello.txt"
        ok = (
            result.stop_reason == "complete"
            and target.exists()
            and "hi" in target.read_text().lower()
        )
        if ok:
            print("\nSMOKE PASSED")
            return 0
        print("\nSMOKE FAILED")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
