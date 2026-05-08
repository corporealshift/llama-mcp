"""Tests for the OpenAI client wrapper with retry logic."""
from unittest.mock import MagicMock

import pytest

from qwen_mcp.openai_client import QwenClient


class _FakeOpenAI:
    """Stand-in for openai.OpenAI; records call count, lets tests script behavior."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_returns_response_on_success():
    fake = _FakeOpenAI([{"choices": [{"message": {"content": "ok"}}]}])
    client = QwenClient.__new__(QwenClient)
    client._client = fake
    client._model = "qwen"
    out = client.chat_completions(messages=[], tools=[])
    assert out == {"choices": [{"message": {"content": "ok"}}]}
    assert fake.calls == 1


def test_retries_on_connection_error_then_succeeds():
    import openai
    err = openai.APIConnectionError(request=MagicMock())
    fake = _FakeOpenAI([err, err, {"choices": []}])
    client = QwenClient.__new__(QwenClient)
    client._client = fake
    client._model = "qwen"
    client._sleep = lambda s: None  # don't actually sleep
    out = client.chat_completions(messages=[], tools=[])
    assert fake.calls == 3
    assert out == {"choices": []}


def test_gives_up_after_three_retries():
    import openai
    err = openai.APIConnectionError(request=MagicMock())
    fake = _FakeOpenAI([err, err, err, err])
    client = QwenClient.__new__(QwenClient)
    client._client = fake
    client._model = "qwen"
    client._sleep = lambda s: None
    with pytest.raises(openai.APIConnectionError):
        client.chat_completions(messages=[], tools=[])
    assert fake.calls == 4  # initial + 3 retries


def test_4xx_not_retried():
    import openai
    err = openai.BadRequestError(
        message="bad", response=MagicMock(status_code=400), body=None
    )
    fake = _FakeOpenAI([err])
    client = QwenClient.__new__(QwenClient)
    client._client = fake
    client._model = "qwen"
    client._sleep = lambda s: None
    with pytest.raises(openai.BadRequestError):
        client.chat_completions(messages=[], tools=[])
    assert fake.calls == 1
