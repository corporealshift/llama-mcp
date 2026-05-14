"""Thin wrapper over the OpenAI client with retry-on-transient logic."""
from __future__ import annotations

import time
from typing import Any

import openai

from llama_mcp.config import Config


class LlamaClient:
    def __init__(self, config: Config) -> None:
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._model = config.model
        self._sleep = time.sleep

    def chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Any:
        backoffs = [1, 2, 4]
        attempt = 0
        while True:
            try:
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                if attempt >= len(backoffs):
                    raise
                self._sleep(backoffs[attempt])
                attempt += 1
            # 4xx errors fall through and propagate without retry.
