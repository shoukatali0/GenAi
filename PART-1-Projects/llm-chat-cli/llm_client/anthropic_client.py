import os
import time
from typing import Iterator, Optional

from anthropic import Anthropic

from .base import ChatResult, LLMClient
from .pricing import estimate_cost


class AnthropicClient(LLMClient):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """Anthropic takes `system` as its own top-level parameter, not as a
        message in the list the way OpenAI does. This is exactly the kind of
        SDK-shape difference your LLMClient interface exists to hide from the
        rest of your code — the CLI never needs to know this quirk exists."""
        system = None
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(m)
        return system, rest

    def stream_chat(self, messages, temperature=0.7, max_tokens=1024) -> Iterator[str]:
        system, rest = self._split_system(messages)
        with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=rest,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def chat(self, messages, temperature=0.7, max_tokens=1024) -> ChatResult:
        system, rest = self._split_system(messages)
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=rest,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        cost = estimate_cost(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=cost,
            latency_s=latency,
            model=self.model,
        )
