tags: [genai/part1, genai/fundamentals]

# 01 — LLM Fundamentals & API Mastery

  

⬅ [[00-START-HERE]]

  

## Why this matters

Every framework (LangChain, LangGraph, agent libraries) is a wrapper around raw API calls. If you understand what's underneath, you debug 10x faster and never get stuck when an abstraction breaks.

  

## Core Concepts (read once)

- **Tokenization**: text → tokens (BPE-style). Token count drives cost AND context limits. Roughly 4 chars ≈ 1 token in English.

- **Context window**: max tokens (input + output combined) a model can handle in one call. Bigger ≠ always better — "lost in the middle" effect means relevant info buried mid-context gets ignored more often.

- **Sampling parameters**:

  - `temperature` — 0 = deterministic/factual, higher = more varied/creative. Use 0–0.3 for extraction/classification, 0.7+ for creative writing.

  - `top_p` (nucleus sampling) — usually leave default, don't tune both temp and top_p at once.

  - `max_tokens` — caps output length AND cost; always set explicitly in production.

- **Message roles**: `system` (persistent instructions/persona), `user`, `assistant`. System prompt is your highest-leverage lever.

- **Streaming**: token-by-token response delivery — required for any chat UI, changes how you write your API call code (async generators).

- **Provider landscape**: OpenAI, Anthropic, Google Gemini (hosted APIs) vs open-source (Llama, Mistral, Qwen) served via Groq/Together/Ollama. Same core concepts, different SDKs and slightly different message formats.

  

## Project: Universal LLM Chat CLI

Build a single Python CLI that talks to **3+ providers** through one interface.

  

**Requirements:**

- Config-driven provider switching (`--provider openai|anthropic|ollama`)

- Streaming output to terminal

- Token counter + estimated cost per call (use provider pricing or `tiktoken` for OpenAI-style counting)

- Conversation history maintained across turns in the same session

- Clean abstraction: a `LLMClient` interface/protocol that each provider implements (this pattern reappears constantly in production code)

  

**Stretch goal:** `--compare` flag that sends the same prompt to all 3 providers and prints outputs side by side with latency + cost for each — your first informal eval tool.

  

**Stack:** Python, `openai`, `anthropic` SDKs, `ollama` (local), `tiktoken`

  

## Resources

- OpenAI API docs — Chat Completions / Responses API

- Anthropic API docs — Messages API

- Ollama docs (for local model serving — also sets up Module 8)

  