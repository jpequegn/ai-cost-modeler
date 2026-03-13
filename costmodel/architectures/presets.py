"""Built-in architecture presets for common AI pipeline patterns.

All token estimates are calibrated for a realistic *code-review* task:
  - System prompt + PR diff + instructions ≈ 5 000–8 000 input tokens
  - Review commentary output ≈ 500–1 200 output tokens
  - Multi-agent workflows add planning / synthesis stages

Usage::

    from costmodel.architectures.presets import (
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    )

    from costmodel.estimator import estimate

    est = estimate(SINGLE_AGENT_SONNET)
    print(f"${est.per_run_usd:.4f} per review")
"""

from __future__ import annotations

from costmodel.models import Architecture, ModelCall, Stage

# ---------------------------------------------------------------------------
# Single-agent presets
# Token estimate basis (code-review task):
#   input  = 5 000 tokens  (system prompt ~500 + PR diff ~3 500 + context ~1 000)
#   output = 600  tokens  (structured review with findings and suggestions)
# ---------------------------------------------------------------------------

SINGLE_AGENT_HAIKU: Architecture = Architecture(
    name="single-agent-haiku",
    description=(
        "Single Haiku agent performing a code review. "
        "Fast and cheap; suitable for style / lint-level feedback."
    ),
    tags=["single-agent", "haiku", "code-review"],
    stages=[
        Stage(
            name="review",
            calls=[
                ModelCall(
                    model="claude-haiku-4-5",
                    input_tokens=5_000,
                    output_tokens=600,
                )
            ],
        )
    ],
)

SINGLE_AGENT_SONNET: Architecture = Architecture(
    name="single-agent-sonnet",
    description=(
        "Single Sonnet agent performing a code review. "
        "Good balance of quality and cost for most teams."
    ),
    tags=["single-agent", "sonnet", "code-review"],
    stages=[
        Stage(
            name="review",
            calls=[
                ModelCall(
                    model="claude-sonnet-4-5",
                    input_tokens=5_000,
                    output_tokens=600,
                )
            ],
        )
    ],
)

SINGLE_AGENT_OPUS: Architecture = Architecture(
    name="single-agent-opus",
    description=(
        "Single Opus agent performing a code review. "
        "Highest quality; best for security-critical or complex changes."
    ),
    tags=["single-agent", "opus", "code-review"],
    stages=[
        Stage(
            name="review",
            calls=[
                ModelCall(
                    model="claude-opus-4-5",
                    input_tokens=5_000,
                    output_tokens=600,
                )
            ],
        )
    ],
)

# ---------------------------------------------------------------------------
# Three-agent parallel Sonnet preset
# Architecture:
#   1. planning  — Sonnet reads the PR and produces a review checklist
#   2. review    — 3 Sonnet agents run *in parallel*, each focusing on a
#                  different dimension (correctness, style, security)
#   3. synthesis — Sonnet merges the three reviews into one final report
# ---------------------------------------------------------------------------

THREE_AGENT_SONNET: Architecture = Architecture(
    name="3-agent-sonnet-parallel",
    description=(
        "Three Sonnet agents reviewing in parallel (correctness / style / security), "
        "preceded by a planning step and followed by a synthesis step. "
        "~3× the cost of a single Sonnet review but higher coverage."
    ),
    tags=["multi-agent", "sonnet", "code-review", "parallel"],
    stages=[
        Stage(
            name="planning",
            calls=[
                ModelCall(
                    model="claude-sonnet-4-5",
                    input_tokens=3_000,   # PR diff + instructions
                    output_tokens=400,    # checklist / decomposition
                )
            ],
        ),
        Stage(
            name="review",
            parallel=True,              # three agents run concurrently
            calls=[
                ModelCall(
                    model="claude-sonnet-4-5",
                    input_tokens=5_000,  # PR diff + specialised system prompt
                    output_tokens=800,   # focused review per dimension
                    repeats=3,           # correctness + style + security
                )
            ],
        ),
        Stage(
            name="synthesis",
            calls=[
                ModelCall(
                    model="claude-sonnet-4-5",
                    input_tokens=6_000,  # three sub-reviews + original diff
                    output_tokens=700,   # unified final report
                )
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Anthropic commercial code-review baseline
# This represents Anthropic's external code-review product priced at a flat
# $25 per review.  The "external" model acts as a placeholder — costs are
# not computed from tokens but noted in the description.
# ---------------------------------------------------------------------------

ANTHROPIC_CODE_REVIEW: Architecture = Architecture(
    name="anthropic-code-review",
    description=(
        "Anthropic external service at -$25/review flat rate. "
        "Token counts are zero because billing is flat-rate, not token-based. "
        "Use this as a $25 baseline when comparing token-based architectures."
    ),
    tags=["baseline", "commercial", "anthropic"],
    stages=[
        Stage(
            name="review",
            calls=[
                ModelCall(
                    model="external",
                    input_tokens=0,
                    output_tokens=0,
                )
            ],
            condition=1.0,
        )
    ],
)
