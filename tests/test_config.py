"""Tests for config loading and WSL2 gateway URL rewriting."""
from unittest.mock import patch

import pytest

from qwen_mcp.config import Config, _rewrite_localhost_to_gateway, load


def test_load_uses_defaults_when_env_unset(monkeypatch):
    for var in [
        "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_API_KEY",
        "QWEN_DEFAULT_MAX_STEPS", "QWEN_DEFAULT_TIMEOUT_SECONDS",
        "QWEN_DEFAULT_MAX_TOKENS_TOTAL", "QWEN_LOG_LEVEL",
    ]:
        monkeypatch.delenv(var, raising=False)

    cfg = load()
    assert cfg.model == "qwen"
    assert cfg.api_key == "sk-no-key"
    assert cfg.default_max_steps == 45
    assert cfg.default_timeout_seconds == 1800
    assert cfg.default_max_tokens_total == 200_000
    assert cfg.log_level == "INFO"


def test_load_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "custom-qwen")
    monkeypatch.setenv("QWEN_DEFAULT_MAX_STEPS", "10")
    monkeypatch.setenv("QWEN_BASE_URL", "http://192.168.1.5:8033/v1")
    cfg = load()
    assert cfg.model == "custom-qwen"
    assert cfg.default_max_steps == 10
    assert cfg.base_url == "http://192.168.1.5:8033/v1"


def test_localhost_rewrites_to_gateway():
    with patch("qwen_mcp.config._wsl2_gateway_ip", return_value="172.20.16.1"):
        result = _rewrite_localhost_to_gateway("http://localhost:8033/v1")
    assert result == "http://172.20.16.1:8033/v1"


def test_127_0_0_1_rewrites_to_gateway():
    with patch("qwen_mcp.config._wsl2_gateway_ip", return_value="172.20.16.1"):
        result = _rewrite_localhost_to_gateway("http://127.0.0.1:8033/v1")
    assert result == "http://172.20.16.1:8033/v1"


def test_other_hosts_pass_through():
    with patch("qwen_mcp.config._wsl2_gateway_ip", return_value="172.20.16.1"):
        result = _rewrite_localhost_to_gateway("http://10.0.0.5:8033/v1")
    assert result == "http://10.0.0.5:8033/v1"


def test_host_docker_internal_passes_through_when_resolvable():
    """If the hostname resolves, leave it alone — let the OS handle DNS."""
    with patch("qwen_mcp.config._dns_resolves", return_value=True):
        result = _rewrite_localhost_to_gateway(
            "http://host.docker.internal:8033/v1"
        )
    assert result == "http://host.docker.internal:8033/v1"


def test_host_docker_internal_falls_back_to_gateway_when_unresolvable():
    with patch("qwen_mcp.config._dns_resolves", return_value=False), \
         patch("qwen_mcp.config._wsl2_gateway_ip", return_value="172.20.16.1"):
        result = _rewrite_localhost_to_gateway(
            "http://host.docker.internal:8033/v1"
        )
    assert result == "http://172.20.16.1:8033/v1"


def test_invalid_max_steps_raises(monkeypatch):
    monkeypatch.setenv("QWEN_DEFAULT_MAX_STEPS", "not-a-number")
    with pytest.raises(ValueError):
        load()
