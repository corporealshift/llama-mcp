"""Server configuration: env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str
    default_max_steps: int
    default_timeout_seconds: int
    default_max_tokens_total: int
    log_level: str


def load() -> Config:
    """Read environment variables and apply defaults."""
    base_url = os.environ.get("LLAMA_BASE_URL", "http://localhost:8033/v1")

    return Config(
        base_url=base_url,
        model=os.environ.get("LLAMA_MODEL", "llama"),
        api_key=os.environ.get("LLAMA_API_KEY", "sk-no-key"),
        default_max_steps=_int_env("LLAMA_DEFAULT_MAX_STEPS", 100),
        default_timeout_seconds=_int_env("LLAMA_DEFAULT_TIMEOUT_SECONDS", 1800),
        default_max_tokens_total=_int_env("LLAMA_DEFAULT_MAX_TOKENS_TOTAL", 200_000),
        log_level=os.environ.get("LLAMA_LOG_LEVEL", "INFO"),
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e
