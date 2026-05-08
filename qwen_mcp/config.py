"""Server configuration: env loading + WSL2 gateway resolution."""
from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


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
    """Read environment variables, apply defaults, rewrite localhost → gateway."""
    raw_url = os.environ.get(
        "QWEN_BASE_URL", "http://host.docker.internal:8033/v1"
    )
    base_url = _rewrite_localhost_to_gateway(raw_url)

    return Config(
        base_url=base_url,
        model=os.environ.get("QWEN_MODEL", "qwen"),
        api_key=os.environ.get("QWEN_API_KEY", "sk-no-key"),
        default_max_steps=_int_env("QWEN_DEFAULT_MAX_STEPS", 45),
        default_timeout_seconds=_int_env("QWEN_DEFAULT_TIMEOUT_SECONDS", 1800),
        default_max_tokens_total=_int_env(
            "QWEN_DEFAULT_MAX_TOKENS_TOTAL", 200_000
        ),
        log_level=os.environ.get("QWEN_LOG_LEVEL", "INFO"),
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _rewrite_localhost_to_gateway(url: str) -> str:
    """Replace loopback/unresolvable hostnames with the WSL2 gateway IP.

    Handles three cases:
    - localhost / 127.0.0.1 / ::1 → always rewrite (these never reach Windows
      from inside WSL2 on Win10).
    - host.docker.internal or any other name → keep if DNS resolves, fall back
      to gateway if not.
    - bare IPs other than loopback → pass through untouched.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return url

    if host in _LOCALHOST_HOSTS:
        return _replace_host(parsed, _wsl2_gateway_ip())

    if _is_ip_literal(host):
        return url

    if _dns_resolves(host):
        return url
    return _replace_host(parsed, _wsl2_gateway_ip())


def _replace_host(parsed, new_host: str) -> str:
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{new_host}{port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _is_ip_literal(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        return False


def _dns_resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


def _wsl2_gateway_ip() -> str:
    """Read the default-route gateway from `ip route` — that's the Windows host."""
    out = subprocess.check_output(
        ["ip", "route", "show", "default"], text=True, timeout=2
    )
    # e.g. "default via 172.20.16.1 dev eth0 ..."
    for token in out.split():
        if token == "via":
            continue
        if _is_ip_literal(token):
            return token
    raise RuntimeError(f"could not parse default gateway from: {out!r}")
