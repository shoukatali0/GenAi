# Day 1 — GenAI Foundations

## What is an LLM?

A Large Language Model (LLM) is a system that predicts the next token based on input text. It does not "think" or "know" facts—it identifies patterns learned during training.

Example:
Input: "I love eating"
Output: "pizza"

---

## Tokens

Tokens are chunks of text used by LLMs instead of full words.

Examples:

* "Hello" → ["Hello"]
* "ChatGPT is amazing" → ["Chat", "G", "PT", " is", " amazing"]

Why tokens matter:

* Cost is based on tokens
* Models have token limits (context window)
* Performance depends on token count

---

## Embeddings

Embeddings convert text into numerical vectors.

Example:
"King" → [0.91, 0.22, ...]
"Queen" → [0.89, 0.25, ...]

Similar meanings have similar vectors.

Use cases:

* Semantic search
* Recommendation systems
* Retrieval-Augmented Generation (RAG)

---

## Context Window

The maximum number of tokens a model can process at once.

Implications:

* Long conversations may lose earlier context
* Important for designing chat systems

---

## Temperature

Controls randomness of output.

* Low (0.2): deterministic, safe
* High (0.9): creative, diverse

---

## Hallucination

When the model generates incorrect or fabricated information.

Reason:

* Predictive nature of LLMs
* Lack of real-time verification

---

## Training vs Inference

Training:

* Model learns from large datasets
* Computationally expensive

Inference:

* Model generates responses to user input
* Happens in real-time

---

## LLM Workflow

User Input → Tokenization → Model Processing → Token Prediction → Output
