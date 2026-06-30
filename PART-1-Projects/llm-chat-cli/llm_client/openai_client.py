import os
import time
from typing import Iterator

from openai import OpenAI

from .base import ChatResult, LLMClient
from .pricing import estimate_cost


class OpenAIClient(LLMClient):
    provider_name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def stream_chat(self, messages, temperature=0.7, max_tokens=1024) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat(self, messages, temperature=0.7, max_tokens=1024) -> ChatResult:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        cost = estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
        return ChatResult(
            text=resp.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_s=latency,
            model=self.model,
        )
