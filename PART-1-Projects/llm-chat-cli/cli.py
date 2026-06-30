#!/usr/bin/env python3
"""
Universal LLM Chat CLI — Module 1 project.

Usage:
  python cli.py --provider openai
  python cli.py --provider anthropic --model claude-haiku-4-5-20251001
  python cli.py --provider ollama
  python cli.py --compare              # one prompt, all providers, side by side
"""
import argparse
import sys
import time
from typing import Optional

from dotenv import load_dotenv

from llm_client import DEFAULT_MODELS, PROVIDERS, estimate_tokens, get_client

load_dotenv()


def chat_loop(provider: str, model: Optional[str], temperature: float, max_tokens: int):
    client = get_client(provider, model)
    print(f"\nChatting with {client.provider_name}/{client.model}  (type 'exit' to quit)\n")

    # System message — your highest-leverage lever, per Module 1 notes.
    history = [{"role": "system", "content": "You are a concise, helpful assistant."}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # Pre-flight token check — concept from Module 1: context windows are
        # finite, and a single huge paste can silently blow your budget.
        est = estimate_tokens(user_input)
        if est > 2000:
            print(f"   (heads up: ~{est} tokens in that message)")

        history.append({"role": "user", "content": user_input})

        print(f"{client.provider_name}: ", end="", flush=True)
        full_text = ""
        start = time.perf_counter()
        for chunk in client.stream_chat(history, temperature=temperature, max_tokens=max_tokens):
            print(chunk, end="", flush=True)
            full_text += chunk
        latency = time.perf_counter() - start
        print(f"\n   [{latency:.2f}s]\n")

        history.append({"role": "assistant", "content": full_text})

        # NOTE: streaming responses generally don't carry token-usage data,
        # so this loop can't show a running $ total the way --compare can.
        # Exercise: get a real running cost total here. One approach —
        # after the stream finishes, fire client.chat() with the same
        # history just to read .input_tokens/.output_tokens off the
        # result (costs one extra non-streamed call, fine for a CLI).


def compare_mode(prompt: str, temperature: float, max_tokens: int):
    print(f'\nComparing all providers on:\n   "{prompt}"\n')
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    results = []
    for provider in PROVIDERS:
        try:
            client = get_client(provider, DEFAULT_MODELS[provider])
            result = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            results.append((provider, result))
        except Exception as e:
            print(f"  {provider} skipped: {e}")

    if not results:
        print("No providers were available — check your .env / Ollama setup.")
        return

    header = f"{'Provider':<12}{'Model':<28}{'Latency':<10}{'Tokens(in/out)':<18}{'Cost':<10}"
    print(header)
    print("-" * len(header))
    for provider, r in results:
        tok = f"{r.input_tokens}/{r.output_tokens}"
        print(f"{provider:<12}{r.model:<28}{r.latency_s:<10.2f}{tok:<18}${r.cost_usd:<10.5f}")

    print()
    for provider, r in results:
        print(f"--- {provider} ---\n{r.text}\n")


def main():
    parser = argparse.ArgumentParser(description="Universal LLM Chat CLI")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    parser.add_argument("--model", default=None, help="Override the default model")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--compare", action="store_true", help="Compare all providers on one prompt")
    args = parser.parse_args()

    if args.compare:
        prompt = input("Prompt to compare across providers: ").strip()
        if not prompt:
            print("No prompt given.")
            sys.exit(1)
        compare_mode(prompt, args.temperature, args.max_tokens)
    else:
        chat_loop(args.provider, args.model, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
