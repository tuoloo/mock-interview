from __future__ import annotations
import json
from typing import Iterator, Protocol, runtime_checkable

import anthropic

from .config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def structured(self, *, prompt: str, schema: dict, system: str | None = None) -> dict: ...
    def stream_text(self, *, prompt: str, system: str | None = None) -> Iterator[str]: ...


class AnthropicClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def structured(self, *, prompt: str, schema: dict, system: str | None = None) -> dict:
        resp = self._client.messages.create(
            model=self._settings.model_id,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self._settings.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            system=system if system is not None else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def stream_text(self, *, prompt: str, system: str | None = None) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._settings.model_id,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": self._settings.effort},
            system=system if system is not None else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
