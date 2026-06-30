from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class ChatResult:
    """What a provider hands back after a non-streaming call."""
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    model: str


class LLMClient(ABC):
    """
    The interface every provider plugs into.

    This is the whole point of Module 1: your CLI code (and later every
    framework you touch — LangChain, LangGraph, agent libraries) talks to
    THIS interface, never directly to OpenAI's SDK or Anthropic's SDK.
    Swap providers = swap the implementation behind this interface, with
    zero changes anywhere else in your code. That's the pattern.
    """

    provider_name: str
    model: str

    @abstractmethod
    def stream_chat(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> Iterator[str]:
        """Yield response text chunk by chunk, as it arrives (live terminal output)."""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> ChatResult:
        """Block until the full response is back; return it with usage + cost + timing.
        Used by --compare mode, where we want clean numbers, not a live stream."""
        raise NotImplementedError
