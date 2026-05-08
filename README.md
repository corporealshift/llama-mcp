# qwen-mcp

MCP server that lets Claude Code delegate code-writing subtasks to a local Qwen
instance running under llama.cpp.

## Install

```bash
pipx install -e /home/kyle/qwen-mcp
```

## Configure Claude Code

Add to `.mcp.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "qwen": {
      "command": "qwen-mcp",
      "env": { "QWEN_BASE_URL": "http://host.docker.internal:8033/v1" }
    }
  }
}
```

See `docs/superpowers/specs/2026-05-08-qwen-mcp-design.md` for design.
