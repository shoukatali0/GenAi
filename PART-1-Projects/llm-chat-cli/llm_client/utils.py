import tiktoken

# cl100k_base is OpenAI's tokenizer family. Claude and Llama use slightly
# different tokenizers under the hood — this is literally the "tokenization"
# concept from your notes in action: token COUNTS genuinely differ by model.
# Good enough for a pre-flight sanity check, not for exact billing.
_encoder = None  # lazy-loaded: tiktoken downloads its BPE file on first use,
                 # so we don't want a network call firing the moment this
                 # module is imported — only when someone actually needs it.


def estimate_tokens(text: str) -> int:
    """Rough token count, used for a pre-flight 'this input is huge' warning."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))