from typing import Optional

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .openai_client import OpenAIClient

PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def get_client(provider: str, model: Optional[str] = None) -> LLMClient:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    cls = PROVIDERS[provider]
    return cls(model=model or DEFAULT_MODELS[provider])
