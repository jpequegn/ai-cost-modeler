"""Model pricing table and cost calculation utilities."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Price table (USD per million tokens)
# Source: Anthropic / OpenAI pricing pages — update as prices change
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude models
    "claude-opus-4-5": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_write_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    },
    "claude-sonnet-4-5": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "cache_write_per_mtok": 3.75,
        "cache_read_per_mtok": 0.30,
    },
    "claude-haiku-4-5": {
        "input_per_mtok": 0.80,
        "output_per_mtok": 4.00,
        "cache_write_per_mtok": 1.00,
        "cache_read_per_mtok": 0.08,
    },
    "claude-opus-4": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_write_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    },
    "claude-sonnet-4": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "cache_write_per_mtok": 3.75,
        "cache_read_per_mtok": 0.30,
    },
    "claude-haiku-4": {
        "input_per_mtok": 0.80,
        "output_per_mtok": 4.00,
        "cache_write_per_mtok": 1.00,
        "cache_read_per_mtok": 0.08,
    },
    "claude-opus-3": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_write_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    },
    "claude-sonnet-3-5": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "cache_write_per_mtok": 3.75,
        "cache_read_per_mtok": 0.30,
    },
    "claude-haiku-3-5": {
        "input_per_mtok": 0.80,
        "output_per_mtok": 4.00,
        "cache_write_per_mtok": 1.00,
        "cache_read_per_mtok": 0.08,
    },
    # OpenAI models
    "gpt-4o": {
        "input_per_mtok": 2.50,
        "output_per_mtok": 10.00,
        "cache_write_per_mtok": 0.00,
        "cache_read_per_mtok": 1.25,
    },
    "gpt-4o-mini": {
        "input_per_mtok": 0.15,
        "output_per_mtok": 0.60,
        "cache_write_per_mtok": 0.00,
        "cache_read_per_mtok": 0.075,
    },
    "o1": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 60.00,
        "cache_write_per_mtok": 0.00,
        "cache_read_per_mtok": 7.50,
    },
    "o3-mini": {
        "input_per_mtok": 1.10,
        "output_per_mtok": 4.40,
        "cache_write_per_mtok": 0.00,
        "cache_read_per_mtok": 0.55,
    },
}

# Short name aliases (for CLI convenience)
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-mini": "gpt-4o-mini",
}

# Task-type heuristic output token estimates
TASK_OUTPUT_TOKENS: dict[str, int] = {
    "code-review": 400,
    "code-generation": 800,
    "summarization": 300,
    "tool-call": 150,
    "planning": 500,
    "chat": 200,
    "question-answering": 250,
    "refactoring": 700,
}

# Task-type heuristic input token estimates (context + system prompt)
TASK_INPUT_TOKENS: dict[str, int] = {
    "code-review": 8000,
    "code-generation": 3000,
    "summarization": 5000,
    "tool-call": 2000,
    "planning": 4000,
    "chat": 1500,
    "question-answering": 2000,
    "refactoring": 6000,
}


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


def cost_for_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute exact cost in USD for one API call.

    Args:
        model: Model name (canonical or alias).
        input_tokens: Number of uncached input tokens.
        output_tokens: Number of output tokens.
        cached_input_tokens: Input tokens served from cache (cheaper).
        cache_write_tokens: Input tokens written to cache (slightly more expensive).

    Returns:
        Cost in USD.
    """
    model = resolve_model(model)
    prices = PRICING[model]
    per_mtok = 1_000_000

    cost = (
        input_tokens * prices["input_per_mtok"] / per_mtok
        + output_tokens * prices["output_per_mtok"] / per_mtok
        + cached_input_tokens * prices["cache_read_per_mtok"] / per_mtok
        + cache_write_tokens * prices["cache_write_per_mtok"] / per_mtok
    )
    return cost


def list_models() -> list[str]:
    """Return all known model names."""
    return list(PRICING.keys())
