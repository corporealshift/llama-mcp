"""Tests for the agent loop using a scripted fake OpenAI client."""
from pathlib import Path
from typing import Any

import pytest

from llama_mcp.agent import AgentResult, run_delegation


class FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content

    def model_dump(self):
        return {"content": self.content, "reasoning_content": self.reasoning_content,
                "tool_calls": [tc.model_dump() for tc in self.tool_calls]}


class FakeToolCall:
    def __init__(self, call_id: str, name: str, args: dict):
        import json
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": json.dumps(args)})()

    def model_dump(self):
        import json
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class FakeChoice:
    def __init__(self, message): self.message = message


class FakeResponse:
    def __init__(self, message, usage=None):
        self.choices = [FakeChoice(message)]
        self.usage = usage


class FakeUsage:
    def __init__(self, total_tokens): self.total_tokens = total_tokens


class ScriptedClient:
    """Returns FakeResponses from a pre-recorded list."""
    def __init__(self, responses): self._responses = list(responses); self.calls = 0
    def chat_completions(self, *, messages, tools, tool_choice="auto"):
        self.calls += 1
        return self._responses.pop(0)


def test_completes_immediately_when_qwen_returns_text(working_dir: Path):
    client = ScriptedClient([
        FakeResponse(FakeMessage(content="Done.", tool_calls=[]),
                     usage=FakeUsage(100)),
    ])
    result = run_delegation(
        client=client,
        working_dir=working_dir,
        task="say done",
        context_hints=[],
        max_steps=10, timeout_seconds=10, max_tokens_total=10_000,
    )
    assert isinstance(result, AgentResult)
    assert result.stop_reason == "complete"
    assert result.result == "Done."
    assert result.steps == 1


def test_executes_tool_call_then_completes(working_dir: Path):
    (working_dir / "x.txt").write_text("hello")
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c1", "read_file", {"path": "x.txt"}),
        ]), usage=FakeUsage(50)),
        FakeResponse(FakeMessage(content="I read x.txt: hello", tool_calls=[]),
                     usage=FakeUsage(100)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="read x.txt",
        context_hints=[], max_steps=10, timeout_seconds=10,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "complete"
    assert "hello" in result.result
    assert result.steps == 2


def test_writes_transcript_file(working_dir: Path):
    client = ScriptedClient([
        FakeResponse(FakeMessage(content="ok"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=10,
        max_tokens_total=10_000,
    )
    transcript_path = Path(result.transcript_path)
    assert transcript_path.exists()
    contents = transcript_path.read_text()
    assert "assistant" in contents


def _looping_tool_call_response():
    """A response that always asks to read a file — would loop forever."""
    return FakeResponse(FakeMessage(content=None, tool_calls=[
        FakeToolCall(f"c{int.from_bytes(b'x','big')}", "list_dir", {"path": "."}),
    ]), usage=FakeUsage(50))


def test_max_steps_enforced(working_dir: Path):
    client = ScriptedClient([_looping_tool_call_response() for _ in range(20)])
    result = run_delegation(
        client=client, working_dir=working_dir, task="loop",
        context_hints=[], max_steps=3, timeout_seconds=60,
        max_tokens_total=1_000_000,
    )
    assert result.stop_reason == "max_steps"
    assert result.steps == 3


def test_token_limit_enforced(working_dir: Path):
    """Total tokens crosses the cap before the loop exits naturally."""
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c1", "list_dir", {"path": "."}),
        ]), usage=FakeUsage(700)),
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c2", "list_dir", {"path": "."}),
        ]), usage=FakeUsage(700)),
        # Should not be reached — token cap trips first.
        FakeResponse(FakeMessage(content="never seen"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=1000,
    )
    assert result.stop_reason == "token_limit"


def test_timeout_enforced(working_dir: Path, monkeypatch):
    """Timeout is checked at top of loop; we fake the clock."""
    import llama_mcp.agent as agent_mod
    fake_now = [1000.0]
    def fake_monotonic():
        fake_now[0] += 0.6
        return fake_now[0]
    monkeypatch.setattr(agent_mod.time, "monotonic", fake_monotonic)

    client = ScriptedClient([_looping_tool_call_response() for _ in range(50)])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=100, timeout_seconds=2,
        max_tokens_total=1_000_000,
    )
    assert result.stop_reason == "timeout"


def test_malformed_tool_args_returned_as_error_to_qwen(working_dir: Path):
    """Qwen sends invalid JSON; loop continues and Qwen self-corrects."""
    bad_call = FakeToolCall("c1", "read_file", {})
    bad_call.function.arguments = "{not json"
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[bad_call]),
                     usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="recovered"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "complete"
    assert result.steps == 2


def test_unknown_tool_name_returned_as_error(working_dir: Path):
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c1", "no_such_tool", {})]),
            usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="recovered"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "complete"


def test_files_changed_accounting(working_dir: Path):
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c1", "write_file",
                         {"path": "a.txt", "content": "hi"}),
        ]), usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="done"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert any(p.endswith("a.txt") for p in result.files_changed)


def test_commands_run_accounting(working_dir: Path):
    client = ScriptedClient([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("c1", "run_command", {"command": "echo hi"}),
        ]), usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="done"), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert "echo hi" in result.commands_run


def test_empty_turn_reprompts_instead_of_reporting_complete(working_dir: Path):
    """An empty message (no content, no tool calls) is not a completion."""
    client = ScriptedClient([
        FakeResponse(FakeMessage(content="", tool_calls=[]), usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="Finished.", tool_calls=[]), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "complete"
    assert result.result == "Finished."
    assert result.steps == 2


def test_tool_call_leaked_into_reasoning_reprompts(working_dir: Path):
    """A <tool_call> left as raw text in reasoning_content is malformed, not done."""
    leaked = FakeMessage(
        content="",
        reasoning_content="I should read it.\n<tool_call>\n<function=read_file>\n"
                          "<parameter=path>\nx.txt\n</parameter>\n</function>\n</tool_call>",
        tool_calls=[],
    )
    client = ScriptedClient([
        FakeResponse(leaked, usage=FakeUsage(10)),
        FakeResponse(FakeMessage(content="Done.", tool_calls=[]), usage=FakeUsage(10)),
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "complete"
    assert result.result == "Done."
    assert result.steps == 2


def test_persistent_malformed_turns_yield_malformed_stop_reason(working_dir: Path):
    """When reprompting keeps failing, the delegation stops with an honest reason."""
    client = ScriptedClient([
        FakeResponse(FakeMessage(content="", tool_calls=[]), usage=FakeUsage(10))
        for _ in range(6)
    ])
    result = run_delegation(
        client=client, working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "malformed"
    assert result.steps == 3


def test_openai_client_error_yields_error_stop_reason(working_dir: Path):
    class BrokenClient:
        def chat_completions(self, **kw): raise RuntimeError("boom")

    result = run_delegation(
        client=BrokenClient(), working_dir=working_dir, task="x",
        context_hints=[], max_steps=10, timeout_seconds=60,
        max_tokens_total=10_000,
    )
    assert result.stop_reason == "error"
    assert "boom" in result.result
