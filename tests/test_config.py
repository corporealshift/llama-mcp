"""Tests for config loading."""
import pytest

from llama_mcp.config import Config, load


def test_load_uses_defaults_when_env_unset(monkeypatch):
    for var in [
        "LLAMA_BASE_URL", "LLAMA_MODEL", "LLAMA_API_KEY",
        "LLAMA_DEFAULT_MAX_STEPS", "LLAMA_DEFAULT_TIMEOUT_SECONDS",
        "LLAMA_DEFAULT_MAX_TOKENS_TOTAL", "LLAMA_LOG_LEVEL",
    ]:
        monkeypatch.delenv(var, raising=False)

    cfg = load()
    assert cfg.base_url == "http://localhost:8033/v1"
    assert cfg.model == "llama"
    assert cfg.api_key == "sk-no-key"
    assert cfg.default_max_steps == 45
    assert cfg.default_timeout_seconds == 1800
    assert cfg.default_max_tokens_total == 200_000
    assert cfg.log_level == "INFO"


def test_load_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("LLAMA_MODEL", "custom-llama")
    monkeypatch.setenv("LLAMA_DEFAULT_MAX_STEPS", "10")
    monkeypatch.setenv("LLAMA_BASE_URL", "http://192.168.1.5:8033/v1")
    cfg = load()
    assert cfg.model == "custom-llama"
    assert cfg.default_max_steps == 10
    assert cfg.base_url == "http://192.168.1.5:8033/v1"


def test_invalid_max_steps_raises(monkeypatch):
    monkeypatch.setenv("LLAMA_DEFAULT_MAX_STEPS", "not-a-number")
    with pytest.raises(ValueError):
        load()
