PRICING = {
    "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
    "gpt-4o": {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-3-5-sonnet-20240620": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "llama3.2": {"input": 0.0, "output": 0.0},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD."""
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * rates["input"]) + (output_tokens * rates["output"])

