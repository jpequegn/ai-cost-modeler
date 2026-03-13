"""Model pricing table and cost calculation utilities."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pricing version — update this whenever prices are manually refreshed
# ---------------------------------------------------------------------------

PRICING_VERSION = "2025-03-12"
"""Date of last manual price update (YYYY-MM-DD)."""

# ---------------------------------------------------------------------------
# Price table (USD per million tokens — MTok)
# Source: Anthropic / OpenAI pricing pages — update as prices change
# Keys: 'input', 'output', 'cache_write', 'cache_read'  (all per MTok)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # ── Anthropic Claude models ──────────────────────────────────────────────
    # claude-opus-4 generation
    "claude-opus-4-6": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    # claude-sonnet-4 generation
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # claude-haiku-4-5
    "claude-haiku-4-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    # Prior claude-opus-4-5 / claude-sonnet-4-5 aliases kept for compatibility
    "claude-opus-4-5": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # Older claude-opus-4 / sonnet-4 / haiku-4 short names
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    # Claude 3.x
    "claude-opus-3": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-3-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-3-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    # ── External / flat-rate services ───────────────────────────────────────
    # Use model="external" for flat-rate services where billing is not
    # token-based (e.g. Anthropic's commercial code-review product).
    # All rates are zero — cost should be annotated in the Architecture
    # description instead.
    "external": {
        "input": 0.00,
        "output": 0.00,
        "cache_write": 0.00,
        "cache_read": 0.00,
    },
    # ── OpenAI models ────────────────────────────────────────────────────────
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        # OpenAI charges no explicit cache_write; cache_read is discounted input
        "cache_write": 0.00,
        "cache_read": 1.25,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "cache_write": 0.00,
        "cache_read": 0.075,
    },
    "o1": {
        "input": 15.00,
        "output": 60.00,
        "cache_write": 0.00,
        "cache_read": 7.50,
    },
    "o3-mini": {
        "input": 1.10,
        "output": 4.40,
        "cache_write": 0.00,
        "cache_read": 0.55,
    },
}

# Short name aliases (for CLI convenience)
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-mini": "gpt-4o-mini",
}

# Task-type heuristic output token estimates (used by estimate_output_tokens)
TASK_OUTPUT_TOKENS: dict[str, int] = {
    "code_generation": 800,
    "code-generation": 800,
    "code_review": 400,
    "code-review": 400,
    "summarization": 300,
    "tool_call": 150,
    "tool-call": 150,
    "planning": 500,
    "chat": 200,
    "question-answering": 250,
    "question_answering": 250,
    "refactoring": 700,
}

# Task-type heuristic input token estimates (context + system prompt)
TASK_INPUT_TOKENS: dict[str, int] = {
    "code_generation": 3000,
    "code-generation": 3000,
    "code_review": 8000,
    "code-review": 8000,
    "summarization": 5000,
    "tool_call": 2000,
    "tool-call": 2000,
    "planning": 4000,
    "chat": 1500,
    "question-answering": 2000,
    "question_answering": 2000,
    "refactoring": 6000,
}


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_model(name: str) -> str:
    """Resolve a short alias or partial name to a canonical model key."""
    if name in PRICING:
        return name
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    # fuzzy match: find first pricing key that contains the name
    lower = name.lower()
    for key in PRICING:
        if lower in key.lower():
            return key
    raise ValueError(
        f"Unknown model '{name}'. Known models: {list(PRICING.keys())} "
        f"or aliases: {list(MODEL_ALIASES.keys())}"
    )


# ---------------------------------------------------------------------------
# Core cost functions
# ---------------------------------------------------------------------------

def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input: int = 0,
) -> float:
    """Compute exact cost in USD for one API call.

    Args:
        model: Model name (canonical key or alias).
        input_tokens: Number of *uncached* input tokens.
        output_tokens: Number of output tokens.
        cached_input: Input tokens served from the prompt cache (cheaper read
                      rate).  These are *not* counted in ``input_tokens``; pass
                      only the tokens that hit the cache.

    Returns:
        Cost in USD rounded to 6 decimal places.

    Example::

        >>> cost_usd('claude-sonnet-4-6', 1000, 500)
        0.010500
        >>> cost_usd('claude-opus-4-6', 10000, 2000)
        0.300000
    """
    model = resolve_model(model)
    prices = PRICING[model]
    per_mtok = 1_000_000

    cost = (
        input_tokens * prices["input"] / per_mtok
        + output_tokens * prices["output"] / per_mtok
        + cached_input * prices["cache_read"] / per_mtok
    )
    return round(cost, 6)


def cost_for_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute exact cost in USD for one API call (extended signature).

    This variant also accounts for cache *write* costs (Anthropic charges a
    small premium for tokens written to the prompt cache).

    Args:
        model: Model name (canonical key or alias).
        input_tokens: Number of uncached input tokens.
        output_tokens: Number of output tokens.
        cached_input_tokens: Input tokens served from cache (cache read rate).
        cache_write_tokens: Input tokens written to cache (cache write rate).

    Returns:
        Cost in USD.
    """
    model = resolve_model(model)
    prices = PRICING[model]
    per_mtok = 1_000_000

    return (
        input_tokens * prices["input"] / per_mtok
        + output_tokens * prices["output"] / per_mtok
        + cached_input_tokens * prices["cache_read"] / per_mtok
        + cache_write_tokens * prices["cache_write"] / per_mtok
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Estimate token count via tiktoken cl100k_base encoding.

    Uses OpenAI's cl100k_base BPE tokeniser (used by GPT-4 / Claude as an
    approximation).  Results match tiktoken directly — there is no rounding or
    heuristic on top of the encoder.

    Args:
        text: The input string to tokenise.

    Returns:
        Number of tokens as an integer.
    """
    import tiktoken  # lazy import — keeps module loadable without tiktoken installed

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Output-token estimation heuristics
# ---------------------------------------------------------------------------

_OUTPUT_HEURISTICS: dict[str, int] = {
    "code_generation": 800,
    "code_review": 400,
    "summarization": 300,
    "tool_call": 150,
    "planning": 500,
    "chat": 200,
}


def estimate_output_tokens(task_type: str) -> int:
    """Return a heuristic estimate of output tokens for a given task type.

    Supported task types and their estimates:

    ==================  ============
    task_type           tokens
    ==================  ============
    code_generation     800
    code_review         400
    summarization       300
    tool_call           150
    planning            500
    chat                200
    ==================  ============

    Hyphenated variants (``code-generation``, ``code-review``, ``tool-call``)
    are also accepted.

    Args:
        task_type: One of the task types listed above.

    Returns:
        Estimated output token count.

    Raises:
        ValueError: If the task_type is not recognised.
    """
    # Normalise hyphen → underscore so both styles work
    key = task_type.replace("-", "_")
    if key in _OUTPUT_HEURISTICS:
        return _OUTPUT_HEURISTICS[key]
    raise ValueError(
        f"Unknown task_type '{task_type}'. "
        f"Supported: {list(_OUTPUT_HEURISTICS.keys())}"
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    """Return all known canonical model names."""
    return list(PRICING.keys())
